import time
import struct
import numpy as np
import mpc
import scipy.spatial as spatial
import sys
import warnings
import sysv_ipc as ipc
import json
from scipy.interpolate import KroghInterpolator
from scipy.interpolate import interp1d
import scipy

warnings.simplefilter('ignore', np.RankWarning)

import matplotlib.pyplot as plt


def yaw_limits(yaw):
  if yaw > np.pi:
      yaw -= 2 * np.pi
  if yaw < -np.pi:
      yaw += 2 * np.pi
  return yaw


t = time.localtime()
logfile = open('data_log_' + str(t.tm_year) + '_' + str(t.tm_mon) + '_' + str(t.tm_mday) + '_' + str(t.tm_hour) + ':' + str(t.tm_min) + ':' + str(t.tm_sec) + '.json', 'w')
logfile = open('data_log.json', 'w')

#MPC parameters
N = 6
prediction_horizon = 40 #m
wheelbase = 2.7 
offset = 120
mpc = mpc.MPC()

throtle = 0.0
steer = 0.0

offset_yaw = 0
#Get reference data
x_ref, y_ref, yaw_ref, ref_vector = [], [], [], []
reference_file = open(sys.argv[1],'r')
line = reference_file.readline()

while len(line) != 0:
    data = eval(line)
    x_ref.append(data['x_ref'])
    y_ref.append(data['y_ref'])
    yaw_ref.append(data['yaw_ref'])
    v_target = data['v_ref']
    ref_vector.append([x_ref[-1], y_ref[-1]])
    line = reference_file.readline()

# Create Tree
waypoints_tree = spatial.cKDTree(ref_vector)

# Shared memory set up
path = "/tmp"
key_recv = ipc.ftok(path, 2333)
key_send = ipc.ftok(path, 2444)
key_send_poly = ipc.ftok(path, 2555)
key_send_tentatives = ipc.ftok(path, 2666)
key_send_tentatives_heading = ipc.ftok(path, 2777)

shm_recv = ipc.SharedMemory(key_recv, 0, 0)
shm_send = ipc.SharedMemory(key_send, 0, 0)
shm_send_poly = ipc.SharedMemory(key_send_poly, 0, 0)
shm_send_tentatives = ipc.SharedMemory(key_send_tentatives, 0, 0)
shm_send_tentatives_heading = ipc.SharedMemory(key_send_tentatives_heading, 0, 0)

shm_recv.attach(0,0)  
shm_send.attach(0,0)  
shm_send_poly.attach(0,0)  
shm_send_tentatives.attach(0,0)  
shm_send_tentatives_heading.attach(0,0)  

sample = 0

try:

  while True:
    start = time.time()

    # Receive Vehicle State
    buf = shm_recv.read(9*8)
    data_recv = struct.unpack('@ddddddddd', buf)
    x_state_vector = data_recv

    if x_state_vector[3] < 3:
       model_type = 'cynematic'
       dt_prediction_horizon = 0.2
       transitory_flag = True
    else:
       model_type = 'dynamic'
       Tc = 0.05
       dt_prediction_horizon_limit = prediction_horizon/(N*x_state_vector[3]) 
       if transitory_flag == True and dt_prediction_horizon < dt_prediction_horizon_limit:
           dt_prediction_horizon = dt_prediction_horizon + Tc*(dt_prediction_horizon_limit - 0.2)/4
       else:
           dt_prediction_horizon = dt_prediction_horizon_limit
           transitory_flag = False

   #if x_state_vector[3] == 0:
   #    dt_prediction_horizon = 0.4
   #else:
   #    dt_prediction_horizon = prediction_horizon/(N*x_state_vector[3]) 
   #print (model_type)



    if x_state_vector[7] == 1.0:
        print('Collision ..................')

    # Receive Obstacle Position
    #buf_obstacle = shm_obstacle_recv.read(2*8)
    #data_recv_obstacle = struct.unpack('@dd', buf_obstacle)

    nearest_neighbour = waypoints_tree.query([x_state_vector[0], x_state_vector[1]], n_jobs=-1)
    index = nearest_neighbour[1]

    # Path planning algorithm
    points_ref = np.transpose(np.array([x_ref[index:index+N+offset], y_ref[index:index+N+offset], np.zeros(N+offset)]))
    yaw_ref = np.arctan2(np.diff(points_ref[:,1]), np.diff(points_ref[:,0]))
    yaw_ref = np.unwrap(yaw_ref)
    yaw_der_ref = np.diff(yaw_ref)

    points_ref_estimated_x = []
    points_ref_estimated_y = []

    if np.mean(np.diff(points_ref[:,0])) != 0.0 and np.mean(np.diff(points_ref[1:,0])) != 0.0 and np.mean(np.diff(points_ref[2:,0])) != 0.0:
      spl_x_estimated_poly = np.polynomial.polynomial.Polynomial.fit(points_ref[:,0], points_ref[:,1], deg=40)
      points_ref_estimated_x = np.transpose(np.array([points_ref[:,0], spl_x_estimated_poly(points_ref[:,0]), np.zeros(N+offset)]))

    if np.mean(np.diff(points_ref[:,1])) != 0.0 and np.mean(np.diff(points_ref[1:,1])) != 0.0 and np.mean(np.diff(points_ref[2:,1])) != 0.0:
      spl_y_estimated_poly = np.polynomial.polynomial.Polynomial.fit(points_ref[:,1], points_ref[:,0], deg=40)
      points_ref_estimated_y = np.transpose(np.array([spl_y_estimated_poly(points_ref[:,1]), points_ref[:,1], np.zeros(N+offset)]))

    x_distance_error = 0
    y_distance_error = 0
    
    if points_ref_estimated_x != [] and points_ref_estimated_y != []:
      for a, b in zip(points_ref, points_ref_estimated_x):
          x_distance_error += np.linalg.norm(a-b)
      for a, b in zip(points_ref, points_ref_estimated_y):
          y_distance_error += np.linalg.norm(a-b)
    elif points_ref_estimated_x == []:
      x_distance_error = 10000
    elif points_ref_estimated_y == []:
      y_distance_error = 10000
    
    sample += 1
    if x_distance_error < y_distance_error:
        #print ('f_x')
        spl_yaw_estimated_poly = np.polynomial.polynomial.Polynomial.fit(points_ref[1:,0], yaw_ref, deg=40)
        #spl_yaw_der_estimated_poly = np.polynomial.polynomial.Polynomial.fit(points_ref[2:,0], yaw_der_ref, deg=16)
        #curvature = spl_yaw_der_estimated_poly(points_ref[3:,0])[0]
        sol_mpc = mpc.opt(x_state_vector[:8], dt_prediction_horizon*np.ones(N), wheelbase, N, spl_x_estimated_poly, spl_yaw_estimated_poly, 'f_x', v_target, model_type)
        points_poly = points_ref_estimated_x
    else:
        #print ('f_y')
        #spl_yaw_estimated_poly = scipy.interpolate.KroghInterpolator(points_ref[1:,1], yaw_ref)
        spl_yaw_estimated_poly = np.polynomial.polynomial.Polynomial.fit(points_ref[1:,1], yaw_ref, deg=40)
        #spl_yaw_der_estimated_poly = np.polynomial.polynomial.Polynomial.fit(points_ref[2:,1], yaw_der_ref, deg=16)
        #curvature = spl_yaw_der_estimated_poly(points_ref[3:,1])[0]
        sol_mpc = mpc.opt(x_state_vector[:8], dt_prediction_horizon*np.ones(N), wheelbase, N, spl_y_estimated_poly, spl_yaw_estimated_poly, 'f_y', v_target, model_type)
        points_poly = points_ref_estimated_y

    no_debug_information = True
    if no_debug_information == False:
      print ('[throtle Steering] ' + str([throtle, steer]))
      print ('Execution time MPC: ' + str(time.time() - start))
      
      print ('X_State_Reference: ' + str([mpc.get_x_target()]))
      print ('Y_State_Reference: ' + str([mpc.get_y_target()]))
      print ('Yaw_State_Reference: ' + str([mpc.get_h_target()]))
      print ('X_State_Vector: ' + str([round(x_state_vector[0],3), round(x_state_vector[1],3), round(x_state_vector[2],3), round(x_state_vector[3],3), round(x_state_vector[4],3)]))
      #print ('Model vector: ' +str(mpc.get_model_state_list()))
      print ('Error d_lateral: ' +str(mpc.get_ed_lateral()))
      print ('Error et: ' +str(mpc.get_ey()))
      print ('Error ev: ' +str(mpc.get_ev()))
      print ('Global Error: ' +str(mpc.get_error()))
      print ('Steering angle: (%3.3f,%3.3f,%3.3f,%3.3f,%3.3f,%3.3f)' % (sol_mpc[0], sol_mpc[1], sol_mpc[2], sol_mpc[3], sol_mpc[4], sol_mpc[5]))
      print ('throtle: ' + str([round(sol_mpc[N+0],3), round(sol_mpc[N+1],3), round(sol_mpc[N+2],3), round(sol_mpc[N+3],3), round(sol_mpc[N+4],3), round(sol_mpc[N+5],3)]))
      print ('----------------------------')
    #print (np.rad2deg(np.diff(sol_mpc[:6])))


    if x_state_vector[3] < 3: 
        sol_mpc = [0, 0, 0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    steer = sol_mpc[0]
    throtle = sol_mpc[N]

    #sol_mpc[N:] = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

    el = mpc.get_ed_lateral()
    ey = mpc.get_ey()
    ev = mpc.get_ev()
    x_state_tentatives = mpc.get_model_state_list()
    type_fun = mpc.get_type_fun()
    slip_angle = mpc.get_slip_angle()
    
    # Datalog information is recorded
    data_to_log = {'x': x_state_vector[0], 'y': x_state_vector[1], 'yaw': x_state_vector[2], 'yaw_ref': mpc.get_h_target()[0], 'type_fun': type_fun, 'v': x_state_vector[3], 'acc': throtle, 'steer_0': sol_mpc[0], 'steer_1': sol_mpc[1], 'steer_2': sol_mpc[2], 'steer_3': sol_mpc[3], 'steer_4': sol_mpc[4], 'steer_5': sol_mpc[5], 'error_lat_0': el[0], 'error_lat_1': el[1], 'error_lat_2': el[2], 'error_lat_3': el[3], 'error_lat_4': el[4], 'error_lat_5': el[5], 'type_lat_error': 0, 'error_yaw': ey[0], 'error_v': ev[0], 'dt': time.time()-start, 'fun_value': mpc.get_sol().fun, 'slip_angle_0': slip_angle[0], 'slip_angle_1': slip_angle[1], 'slip_angle_2': slip_angle[2], 'slip_angle_3': slip_angle[3], 'slip_angle_4': slip_angle[4], 'slip_angle_5': slip_angle[5]}
    json.dump(data_to_log, logfile)
    logfile.write('\n')

    
    # Send Control actions to Car with Shared Memory 
    data_send = struct.pack('@ddddddddddddd', sol_mpc[0], sol_mpc[1], sol_mpc[2], sol_mpc[3], sol_mpc[4], sol_mpc[5], sol_mpc[N], sol_mpc[N+1],
                             sol_mpc[N+2], sol_mpc[N+3], sol_mpc[N+4], sol_mpc[N+5], index)
    shm_send.write(data_send)

    data_send = struct.pack('@dddddddddddddddd', points_poly[0][0], points_poly[0][1], points_poly[1][0], points_poly[1][1], points_poly[2][0], points_poly[2][1], points_poly[3][0], points_poly[3][1], points_poly[4][0], points_poly[4][1], points_poly[5][0], points_poly[5][1], points_poly[6][0], points_poly[6][1], points_poly[7][0], points_poly[7][1])
    shm_send_poly.write(data_send)

    data_send = struct.pack('@dddddddddddd', x_state_tentatives[0][0], x_state_tentatives[0][1], x_state_tentatives[1][0], x_state_tentatives[1][1], x_state_tentatives[2][0], x_state_tentatives[2][1], x_state_tentatives[3][0], x_state_tentatives[3][1], x_state_tentatives[4][0], x_state_tentatives[4][1], x_state_tentatives[5][0], x_state_tentatives[5][1])
    shm_send_tentatives.write(data_send)

    data_send = struct.pack('@dddddd', x_state_tentatives[0][2], x_state_tentatives[1][2], x_state_tentatives[2][2], x_state_tentatives[3][2], x_state_tentatives[4][2], x_state_tentatives[5][2])
    shm_send_tentatives_heading.write(data_send)

except KeyboardInterrupt:
    print ('MPC controller has been stoped')
    data_send = struct.pack('@ddddddddddddd', 0, 0,0,0,0,0,0,0,0,0,0,0,-1)
    shm_send.write(data_send)
    shm_recv.detach()
    shm_send.detach()
    shm_send_poly.detach()
    shm_send_tentatives.detach()
    shm_send_tentatives_heading.detach()

    logfile.close()
    sys.exit()
