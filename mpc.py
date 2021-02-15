from transforms3d.euler import euler2mat, mat2euler
from scipy import optimize
import numpy as np
from warnings import filterwarnings
from sympy import *
import matplotlib.pyplot as plt
from sympy.geometry import *
import model_carla as model
import json
import utils
import time
import transforms3d.euler
import scipy.spatial as spatial
from scipy.optimize import NonlinearConstraint

class MPC:
    def __init__(self):
       print ('MPC process new' )
       self.cont = 0
       self.x_state_tentatives = None
       self.x_targets = None
       self.y_targets = None
       self.h_targets = None
       self.ey = None
       self.ed_lateral = [0, 0, 0, 0, 0, 0]
       self.type_fun = None
       self.obstacle_pos = None
       self.dist_obs_min_1 = None
       self.dist_obs_min_2 = None
       self.dist_obs_min_3 = None
       self.dist_obs_min_4 = None
       self.dist_obs_min_5 = None
       self.dist_obs_min_6 = -0.1 #5
       self.dist_obs_max = 0.1
       self.heading_error_value = 0.02
       clean_file = open('output.json', 'w')
       self.error = 0
       self.teta_vf = None
       self.sol = None
       self.vy = 0
       self.slip_angle = 0
       self.slip_angle_list = [0, 0, 0, 0, 0, 0]
       self.vx_list = [0, 0, 0, 0, 0, 0]
       self.ay = 0
       self.ay_list = [0, 0, 0, 0, 0, 0]
       self.ax = 0
       self.ax_list = [0, 0, 0, 0, 0, 0]

       # Obstacles avoidance code
       self.list_obstacle_pos = []
       self.obstacle_flag = False
       if self.obstacle_flag == True:
         #self.list_obstacle_pos.append([-88, 90, np.deg2rad(180-90)]) #OBSTACLE 0 (recta 1)
         #self.list_obstacle_pos.append([-88, 165, np.deg2rad(180-75)]) #OBSTACLE 1 (curva 1)
         #self.list_obstacle_pos.append([-9, 207.3, np.deg2rad(180-0)]) #OBSTACLE 2 (recta 2)
         
         self.list_obstacle_pos.append([41, 207.70, np.deg2rad(180-0)]) #OBSTACLE 3
         self.list_obstacle_pos.append([230, 175, np.deg2rad(180-(-42))]) #OBSTACLE 4
         self.list_obstacle_pos.append([248.6, -113, np.deg2rad(180-(-89))]) #OBSTACLE 5

    def get_sol(self):
        return self.sol

    def get_model_state_list(self):
        return self.x_state_tentatives

    def get_x_target(self):
        return self.x_targets

    def get_y_target(self):
        return self.y_targets

    def get_h_target(self):
        return self.h_targets

    def get_ey(self):
        return self.ey

    def get_ev(self):
        return self.ev

    def get_ed_lateral(self):
        return self.ed_lateral

    def get_error(self):
        return self.error

    def get_slip_angle(self):
        return self.slip_angle_list

    def get_teta_vf(self):
        return self.teta_vf

    def get_acc_y(self):
        return self.ay   

    def get_acc_x(self):
        return self.ax   

    def get_v_y(self):
        return self.vy                     

    def get_vx_list(self):
        return self.vx_list                     

    def get_ax_list(self):
        return self.ax_list                     

    def get_ay_list(self):
        return self.ay_list                     

    def get_type_fun(self):
        return self.type_fun

    def obstacle_lateral_error(self, x_state, label, poly_ref, poly_angle_ref):
      if label == 'f_x':
        heading = poly_angle_ref(x_state[0])
        ed_lateral = (x_state[1] - poly_ref(x_state[0]))*np.cos(heading)
      else:
        heading = poly_angle_ref(x_state[1])
        ed_lateral = -(x_state[0] - poly_ref(x_state[1]))*np.cos(heading - np.pi/2)

      if ed_lateral > 20:
          ed_lateral = 20
      elif ed_lateral < -20:
          ed_lateral = -20

      return ed_lateral

    def heading_error(self, x_state, label, poly_ref, poly_angle_ref):
      if label == 'f_x':
        heading = poly_angle_ref(x_state[0])
      else:
        heading = poly_angle_ref(x_state[1])
      ey = model.normalize_angle(x_state[2] - heading)

      return ey

    def obstacle_longitudinal_error(self, veh_states, obs_pos):
      rotation_matrix = transforms3d.euler.euler2mat(0, 0, veh_states[2], axes='sxyz')
      rotation_matrix = np.linalg.inv(rotation_matrix)
      obstacle_referenced_to_vehicle = np.dot(rotation_matrix, np.asarray([obs_pos[0], obs_pos[1], 0]) - np.asarray([veh_states[0], veh_states[1], 0]))

      return obstacle_referenced_to_vehicle[0]

    def mpc_process(self, x, *args):
      self.x_state_tentatives = [[args[0][0], args[0][1], args[0][2], args[0][3], args[0][4], args[0][5], args[0][6], self.vy, self.slip_angle, self.ax, self.ay]]
      vx = args[0][3]

      dt_list = args[1]
      lf = args[2]
      N = args[3]
      poly_ref = args[4]
      poly_angle_ref = args[5]
      label = args[6]
      v_target = args[7]*np.ones(N+1) # m/s
      model_type = args[8] 

      for i in range(N):
        steering_angle = x[i]
        throttle = x[i+N] 
        dt = dt_list[i]
        self.x_state_tentatives = np.append(self.x_state_tentatives, [model.move_2(self.x_state_tentatives[-1], dt, [steering_angle, throttle], lf, model_type)], axis=0)
      
      #print (self.x_state_tentatives)
      self.vy = self.x_state_tentatives[1,7]
      self.slip_angle_list = self.x_state_tentatives[1:,8]
      self.vx_list = self.x_state_tentatives[0:,3]
      self.ax_list = self.x_state_tentatives[1:,9]
      self.ay_list = self.x_state_tentatives[1:,10]

      if label == 'f_x':
        self.type_fun = 0

        self.x_targets = self.x_state_tentatives[1:,0]
        self.y_targets = poly_ref(self.x_targets)
        self.h_targets = poly_angle_ref(self.x_targets)

        for i in range(len(self.h_targets)):
            self.h_targets[i] = model.normalize_angle(self.h_targets[i]) 

        #self.ed_lateral = [[(self.x_state_tentatives[1,1] - poly_ref(self.x_state_tentatives[1,0]))*np.cos(self.h_targets[0])]]
        #for x_state_elem, y_state_elem, h_elem in zip(self.x_state_tentatives[2:,0], self.x_state_tentatives[2:,1], self.h_targets[1:]):
        #    self.ed_lateral = np.append(self.ed_lateral, [[(y_state_elem - poly_ref(x_state_elem))*np.cos(h_elem)]])

      elif label == 'f_y':
        self.type_fun = 1
        self.y_targets = self.x_state_tentatives[1:,1]
        self.x_targets = poly_ref(self.y_targets)
        self.h_targets = poly_angle_ref(self.y_targets)

        for i in range(len(self.h_targets)):
            self.h_targets[i] = model.normalize_angle(self.h_targets[i])


        #self.ed_lateral = [[-(self.x_state_tentatives[1,0] - poly_ref(self.x_state_tentatives[1,1]))*np.cos(self.h_targets[0] - np.pi/2)]]
        #for x_state_elem, y_state_elem, h_elem in zip(self.x_state_tentatives[2:,0], self.x_state_tentatives[2:,1], self.h_targets[1:]):
        #    self.ed_lateral = np.append(self.ed_lateral, [[-(x_state_elem - poly_ref(y_state_elem))*np.cos(h_elem - np.pi/2)]])
        
      self.ed_lateral = [[self.obstacle_lateral_error(self.x_state_tentatives[1,:], label, poly_ref, poly_angle_ref)]]
      for x_state_elem in self.x_state_tentatives[2:,:]:
          self.ed_lateral = np.append(self.ed_lateral, [[self.obstacle_lateral_error(x_state_elem, label, poly_ref, poly_angle_ref)]])

      self.ey = [[self.heading_error(self.x_state_tentatives[1,:], label, poly_ref, poly_angle_ref)]]
      for x_state_elem in self.x_state_tentatives[2:,:]:
          self.ey = np.append(self.ey, [[self.obstacle_lateral_error(x_state_elem, label, poly_ref, poly_angle_ref)]])

      #self.ey = [[model.normalize_angle(self.x_state_tentatives[1,2] - self.h_targets[0])]]
      #for t_target_elem, tt_elem in zip(self.h_targets[1:], self.x_state_tentatives[2:,2]):
      #    self.ey = np.append(self.ey, [[model.normalize_angle(tt_elem - t_target_elem)]])

      self.ev = [[v_target[0] - self.x_state_tentatives[0,3]]]
      for v_target_elem, vt_elem in zip(v_target[1:], self.x_state_tentatives[1:,3]):
          self.ev = np.append(self.ev, [[v_target_elem - vt_elem]])

      self.teta_vf = self.x_state_tentatives[1:,6]
      #self.ay = self.x_state_tentatives[1:,]
      self.error = 0

      a = 100000000000

      for slip_angle_elem in self.slip_angle_list:
          self.error += a*np.linalg.norm(slip_angle_elem)

      for ey_elem in self.ey:
          self.error += 60*a*np.linalg.norm(ey_elem)

      for ed_lateral_elem in self.ed_lateral:
          self.error += 20*a*np.linalg.norm(ed_lateral_elem)

      for ev_elem in self.ev:
          self.error += a*np.linalg.norm(ev_elem)

      #for diff_elem in np.diff(x[:N]): #Steer
      for diff_elem in np.diff(self.x_state_tentatives[:,4]): #Steer
            self.error += 1000*a*np.linalg.norm(diff_elem)  #35

      for diff_elem in np.diff(x[N:]):  #throttle
          self.error += a*np.linalg.norm(diff_elem)

      return self.error 

            
    def opt(self, x_state, dt, wheelbase, N, poly_ref, poly_angle_ref, label, v_target, model_type):

      vx = x_state[3]
      steer = x_state[4]
      throttle = x_state[7]

      acc_contraint = 1   #m/s^2 
      steer_limit = np.deg2rad(60)

      throttle_increment = 0.01
      
      self.max_slip_angle = np.deg2rad(2.5)
      increment_sterring = steer_limit
      self.min_vx = v_target 
      self.max_vx = v_target + 0.5

      #Obstacles
      if self.obstacle_flag == True:
        self.max_slip_angle = np.deg2rad(2.5)
        distance_to_obstacle = []
        for obstacle in self.list_obstacle_pos:
        	distance_to_obstacle.append(np.sqrt((x_state[0] - obstacle[0])**2 + (x_state[1] - obstacle[1])**2))
        
        self.obstacle_pos = self.list_obstacle_pos[distance_to_obstacle.index(min(distance_to_obstacle))]
        distance_to_obstacle = self.obstacle_longitudinal_error(x_state, self.obstacle_pos)
              
        offset_slip_angle_rstriction = 5
        distance_threshold = 10
        
        if (-5 <= np.asarray(distance_to_obstacle) < distance_threshold).any(): #5m after overtaking, and 20m before overtaking the obstacle
            if distance_to_obstacle < 10:
              avoidance_distance = 3
            else:
              avoidance_distance = 3 + (3*10/(distance_threshold-10)) - (3/(distance_threshold-10))*distance_to_obstacle
        
            self.dist_obs_min_6 = avoidance_distance
            self.dist_obs_max = avoidance_distance + .05 #m 
        else:
            at = 0.05
            time_to_came_back = 0.25
            self.dist_obs_max = self.dist_obs_max - (3/time_to_came_back)*at
            if self.dist_obs_max <= 0.1: 
              self.dist_obs_max = 0.1 #m 
            self.dist_obs_min_6 = -self.dist_obs_max #m 

        print('Longitudinal distance: ' + str(distance_to_obstacle))

      cons = (NonlinearConstraint(lambda x: x[0],  -steer_limit, steer_limit),
              NonlinearConstraint(lambda x: x[1],  -steer_limit, steer_limit),
              NonlinearConstraint(lambda x: x[2],  -steer_limit, steer_limit),
              NonlinearConstraint(lambda x: x[3],  -steer_limit, steer_limit),
              NonlinearConstraint(lambda x: x[4],  -steer_limit, steer_limit),
              NonlinearConstraint(lambda x: x[5],  -steer_limit, steer_limit),
              NonlinearConstraint(lambda x: x[6],  0, acc_contraint),
              NonlinearConstraint(lambda x: x[7],  0, acc_contraint),
              NonlinearConstraint(lambda x: x[8],  0, acc_contraint),
              NonlinearConstraint(lambda x: x[9],  0, acc_contraint),
              NonlinearConstraint(lambda x: x[10], 0, acc_contraint),
              NonlinearConstraint(lambda x: x[11], 0, acc_contraint),

             #NonlinearConstraint(lambda x: (x[0] - steer), -increment_sterring, increment_sterring),
             #NonlinearConstraint(lambda x: (x[1] - x[0]),  -increment_sterring, increment_sterring),
             #NonlinearConstraint(lambda x: (x[2] - x[1]),  -increment_sterring, increment_sterring),
             #NonlinearConstraint(lambda x: (x[3] - x[2]),  -increment_sterring, increment_sterring),
             #NonlinearConstraint(lambda x: (x[4] - x[3]),  -increment_sterring, increment_sterring),
             #NonlinearConstraint(lambda x: (x[5] - x[4]),  -increment_sterring, increment_sterring),

             #NonlinearConstraint(lambda x: (x[6] - throttle), -throttle_increment, throttle_increment),
             #NonlinearConstraint(lambda x: (x[7] - x[6]),     -throttle_increment, throttle_increment),
             #NonlinearConstraint(lambda x: (x[8] - x[7]),     -throttle_increment, throttle_increment),
             #NonlinearConstraint(lambda x: (x[9] - x[8]),     -throttle_increment, throttle_increment),
             #NonlinearConstraint(lambda x: (x[10] - x[9]),    -throttle_increment, throttle_increment),
             #NonlinearConstraint(lambda x: (x[11] - x[10]),   -throttle_increment, throttle_increment),

              NonlinearConstraint(lambda x: model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type)[3], self.min_vx, self.max_vx),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type)[3], self.min_vx, self.max_vx),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type)[3], self.min_vx, self.max_vx),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type)[3], self.min_vx, self.max_vx),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type)[3], self.min_vx, self.max_vx),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type), dt[5], [x[5], x[11]], 2.7, model_type)[3], self.min_vx, self.max_vx),


              NonlinearConstraint(lambda x: model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type)[8], -self.max_slip_angle, self.max_slip_angle),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type)[8], -self.max_slip_angle, self.max_slip_angle),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type)[8], -self.max_slip_angle, self.max_slip_angle),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type)[8], -self.max_slip_angle, self.max_slip_angle),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type)[8], -self.max_slip_angle, self.max_slip_angle),
              NonlinearConstraint(lambda x: model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type), dt[5], [x[5], x[11]], 2.7, model_type)[8], -self.max_slip_angle, self.max_slip_angle),

              #NonlinearConstraint(lambda x: self.heading_error(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), label, poly_ref, poly_angle_ref), -self.heading_error_value, self.heading_error_value),
              #NonlinearConstraint(lambda x: self.heading_error(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), label, poly_ref, poly_angle_ref), -self.heading_error_value, self.heading_error_value),
              #NonlinearConstraint(lambda x: self.heading_error(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), label, poly_ref, poly_angle_ref), -self.heading_error_value, self.heading_error_value),
              #NonlinearConstraint(lambda x: (self.heading_error(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), label, poly_ref, poly_angle_ref)), -self.heading_error_value, self.heading_error_value),
              #NonlinearConstraint(lambda x: (self.heading_error(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type), label, poly_ref, poly_angle_ref)), -self.heading_error_value, self.heading_error_value),
              NonlinearConstraint(lambda x: -(self.heading_error(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type), dt[5], [x[5], x[11]], 2.7, model_type), label, poly_ref, poly_angle_ref)), -self.heading_error_value, self.heading_error_value),

              #NonlinearConstraint(lambda x: self.obstacle_lateral_error(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), label, poly_ref, poly_angle_ref), self.dist_obs_min_6, self.dist_obs_max),
              #NonlinearConstraint(lambda x: self.obstacle_lateral_error(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), label, poly_ref, poly_angle_ref), self.dist_obs_min_6, self.dist_obs_max),
              #NonlinearConstraint(lambda x: self.obstacle_lateral_error(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), label, poly_ref, poly_angle_ref), self.dist_obs_min_6, self.dist_obs_max),
              #NonlinearConstraint(lambda x: (self.obstacle_lateral_error(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), label, poly_ref, poly_angle_ref)), self.dist_obs_min_6, self.dist_obs_max),
              #NonlinearConstraint(lambda x: (self.obstacle_lateral_error(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type), label, poly_ref, poly_angle_ref)), self.dist_obs_min_6, self.dist_obs_max),
              NonlinearConstraint(lambda x: -(self.obstacle_lateral_error(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [x[0], x[6]], 2.7, model_type), dt[1], [x[1], x[7]], 2.7, model_type), dt[2], [x[2], x[8]], 2.7, model_type), dt[3], [x[3], x[9]], 2.7, model_type), dt[4], [x[4], x[10]], 2.7, model_type), dt[5], [x[5], x[11]], 2.7, model_type), label, poly_ref, poly_angle_ref)), self.dist_obs_min_6, self.dist_obs_max))
              

      if model_type == 'cynematic':
        x0 = 0*np.ones(N)
        x0 = np.append(x0, 0.1*np.ones(N))
      else:
        x0 = self.sol.x

      iterations = 500
      #options = {'maxiter': iterations, 'disp': True}
      options = {'maxiter': iterations, 'disp': False, 'tol':1e-3}

      self.sol = optimize.minimize(self.mpc_process, args=(x_state, dt, wheelbase, N, poly_ref, poly_angle_ref, label, v_target, model_type), x0=x0, method='COBYLA', options=options, constraints=cons)

      #print (self.get_model_state_list()[:,6]) #Acc x
      #print ('Steering Control: ' + str(self.sol.x[:N])) #Vx
      #print ('Steering: ' + str(self.get_model_state_list()[:,4])) #Vx
      #print (str(self.get_model_state_list()[:2,:])) #Vx
      #print ('Teta: ' + str(self.get_model_state_list()[:,7])) #Vx

      #print ('Lateral distance: ' + str(self.obstacle_lateral_error(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [self.sol.x[0], self.sol.x[6]], 2.7, model_type), label, poly_ref, poly_angle_ref)))
      #print ('Slip angle: ' + str(model.move_2(np.concatenate((x_state[0:7], [self.vy], [self.slip_angle], [self.ax], [self.ay])), dt[0], [self.sol.x[0], self.sol.x[6]], 2.7, model_type)[8]))
      #print ('Speed: ' + str(self.vx_list))
      #print ('AccX: ' + str(self.ax_list))
      #print ('AccY: ' + str(self.ay_list))
      #print ('Throttle: ' + str(self.sol.x[N:]))
      #print('Number of iterations -----> '+str(self.cont))
      
      return self.sol.x

