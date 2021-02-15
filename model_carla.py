import numpy as np
from math import tan, sin, cos, sqrt

def move_with_acc(x, dt, u, wheelbase, debug=False):
    steering_angle = u[0]
    x_acc_veh = u[1]

    dist = (0.5*x_acc_veh*(dt**2)) + (x[3]*dt)

    hdg = x[2]
    b=0.2

    if abs(steering_angle) > np.deg2rad(0.0000005): # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase/tan(steering_angle) # radius
        

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        ret =  x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta, (x_acc_veh-b*x[3])*dt])
        ret[2] = normalize_angle(ret[2])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret
    else: # moving in straight line
        ret = x + np.array([dist*cos(hdg), dist*sin(hdg), 0, (x_acc_veh-b*x[3])*dt])
        ret[2] = normalize_angle(ret[2])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        return ret

def move_1(x, dt, u, wheelbase, debug=False):
    steering_angle = u[0]
    x_acc_veh = u[1]

    hdg = x[2]
    vx = x_acc_veh*dt + x[3]
    steering_angle_1 = x[4] 

    b=0.2

    beta = np.arctan(tan(steering_angle)/2)
    yaw =  (vx*dt*np.cos(beta)/wheelbase)*tan(steering_angle)
    
    ret =  x + np.array([vx*dt*np.cos(hdg), vx*dt*np.sin(hdg), yaw, (x_acc_veh-b*x[3])*dt, 0])
    ret[2] = normalize_angle(ret[2])
    if ret[3] < 0.0: #Velocity condition
        ret[3] = 0.0
    ret[4] = steering_angle
    return ret

def move_2(x, dt, u, wheelbase, model_type, debug=False):

    wheel_angle = u[0]
    throttle = u[1]
    if throttle < 0:
        throttle = 0

    hdg = x[2]
    yaw_rate = x[6] 

    m = 1800.0  # Car mass
    Iz = 2800.0
    lf = 1.2
    lr = 1.5
    Cf = 2000.0 
    Cr = 2000.0

    Ie = 1
    Iw = 4.1
    gr = 2.6
    rw = 0.37
    Temax = 450
    cr = 100
    #Je = (Ie + Iw*gr**2 + m*reff**2*gr**2)

    vx = x[3]
    if vx == 0:
        vx = 0.0001

    ay = -x[5] - (vx*yaw_rate)


    if model_type == 'dynamic':
      #ax = reff*gr*((throttle*Temax - gr*reff*cr*abs(vx)))/Je
      #print (ax)
      a = 0.8  #5ms
      #a = 2.9  #10ms
      #a = 5.5  #15ms
      ax = a*1.5*5*throttle - a*.08*abs(vx) - 0.09*vx**2 # 5ms
      #ax = 2*1.5*5*throttle - .1*abs(vx) - 0.075*vx**2 # 10ms
      #ax = 2*1.5*5*throttle - .1*abs(vx) - 0.070*vx**2 # 11ms
      #ax = 2*1.5*5*throttle - .1*abs(vx) - 0.060*vx**2 # 13ms
      #ax = 2*1.5*5*throttle - .1*abs(vx) - 0.03*vx**2 # 17ms
      #ax = 400*throttle*gr/(m*rw) - 0.05*abs(vx)
      xveh = 0.5*ax*dt**2 + vx*dt
      if xveh < 0:
          xveh = 0

      vy = x[7]

      #teta_vf = np.arctan2((vy + lf*yaw_rate),vx)
      #teta_vr = np.arctan2((vy - lr*yaw_rate),vx)
      teta_vf = (vy + lf*yaw_rate)/vx
      teta_vr = (vy - lr*yaw_rate)/vx

      Fyf = 2*Cf*(wheel_angle - teta_vf)
      Fyr = 2*Cr*(-teta_vr)

      ay_1 = (Fyf + Fyr)/m
      yveh = 0.5*ay_1*dt**2 + vy*dt
  
      acc_yaw = (lf*Fyf - lr*Fyr)/Iz
      yaw_rate = (acc_yaw - 0.01*yaw_rate)*dt + yaw_rate

      desplazamiento = np.sqrt(xveh**2 + yveh**2)
      ret =  x + np.array([desplazamiento*np.cos(hdg), desplazamiento*np.sin(hdg), -yaw_rate*dt, ax*dt, 0, 0, 0, ay_1*dt, 0, 0, 0])

      ret[4] = wheel_angle 
      ret[5] = -ay_1  - (vx*yaw_rate)
      ret[6] = yaw_rate 
      ret[8] = wheel_angle - teta_vf
      ret[9] = ax
      ret[10] = ay_1

    else:
      #ax = 1.5*throttle - .1*abs(vx)
      a = 0.8  #5ms
      ax = a*1.5*5*throttle - a*.08*abs(vx) - 0.1*vx**2 # 5ms
      beta = np.arctan(lf*tan(-wheel_angle)/wheelbase)
      yaw = (vx*dt*np.cos(beta)/wheelbase)*tan(-wheel_angle)
      
      ret =  x + np.array([vx*dt*np.cos(hdg), vx*dt*np.sin(hdg), yaw, ax*dt, 0, 0, 0, 0, 0, 0, 0])

      ret[4] = wheel_angle 
      ret[5] = 0
      ret[6] = yaw_rate 
      ret[7] = 0 
      ret[8] = 0
      ret[9] = ax
      ret[10] = 0

    ret[2] = normalize_angle(ret[2])
    if ret[3] < 0.0: #Velocity condition
        ret[3] = 0.0
    return ret

def torque_to_steer(torque, steer_1):
    tau = 4 
    k = 1 
    steering_angle = (k*torque + steer_1)/(tau + 1)
        
    return steering_angle

def move_with_acc_new(x, dt, u, wheelbase, debug=False):
    torque_wheel = u[0]
    x_acc_veh = u[1]

    dist = (0.5*x_acc_veh*(dt**2)) + (x[3]*dt)
    steering_angle_1 = x[4] 

    steering_angle = torque_to_steer(torque_wheel,steering_angle_1)

    hdg = x[2]
    b=0.2

    if abs(steering_angle) > np.deg2rad(0.0000005): # is robot turning?
        beta = (dist / wheelbase) * tan(steering_angle)
        r = wheelbase/tan(steering_angle) # radius
        

        sinh, sinhb = sin(hdg), sin(hdg + beta)
        cosh, coshb = cos(hdg), cos(hdg + beta)
        ret =  x + np.array([-r*sinh + r*sinhb, r*cosh - r*coshb, beta, (x_acc_veh-b*x[3])*dt, 0])
        ret[2] = normalize_angle(ret[2])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        ret[4] = steering_angle
        return ret
    else: # moving in straight line
        ret = x + np.array([dist*cos(hdg), dist*sin(hdg), 0, (x_acc_veh-b*x[3])*dt, 0])
        ret[2] = normalize_angle(ret[2])
        if ret[3] < 0.0: #Velocity condition
            ret[3] = 0.0
        ret[4] = steering_angle
        return ret

def normalize_angle(x):
  x = x % (2 * np.pi)  # force in range [0, 2 pi)
  if x > np.pi:  # move to [-pi, pi)
    x -= 2 * np.pi
  return x

def normalize_angle_list(x_list):

  x_list_normalize = []

  for x in x_list:
    x = x % (2 * np.pi)  # force in range [0, 2 pi)
    if x > np.pi:  # move to [-pi, pi)
      x -= 2 * np.pi
    x_list_normalize.append(x)

  return x_list_normalize

