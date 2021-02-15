import transforms3d.euler
import numpy as np 

def vector_to_matrix_pose(vector_pose):
  matrix_pose = transforms3d.euler.euler2mat(vector_pose['roll'], vector_pose['pitch'], vector_pose['yaw'], axes='sxyz')
  matrix_pose = np.insert(matrix_pose, 3, [vector_pose['x'], vector_pose['y'], vector_pose['z']], axis=1)
  matrix_pose = np.insert(matrix_pose, 3, [0, 0, 0, 1], axis=0)

  return matrix_pose


def matrix_to_vector_pose(matrix_pose):
  rotation_pose = transforms3d.euler.mat2euler(matrix_pose[0:3, 0:3], axes='sxyz')
  traslation_pose = [matrix_pose[0][3], matrix_pose[1][3], matrix_pose[2][3]]

  return {'x': traslation_pose[0], 'y': traslation_pose[1], 'z': traslation_pose[2],
          'roll': rotation_pose[0], 'pitch': rotation_pose[1], 'yaw': rotation_pose[2]}

  return np.concatenate((traslation_pose, rotation_pose), axis=0)
