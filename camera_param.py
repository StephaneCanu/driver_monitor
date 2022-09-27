import numpy as np


eon_focal_length = 910.0  # pixels
eon_dcam_focal_length = 860.0  # pixels

webcam_focal_length = 908.0/1.5  # pixels

eon_intrinsics = np.array([
  [eon_focal_length,   0.,   1164/2.],
  [0.,  eon_focal_length,  874/2.],
  [0.,    0.,     1.]])

eon_dcam_intrinsics = np.array([
  [eon_dcam_focal_length,   0,   1152/2.],
  [0,  eon_dcam_focal_length,  864/2.],
  [0,    0,     1]])

# webcam_intrinsics = np.array([
#   [webcam_focal_length,   0.,   1280/2],
#   [0.,  webcam_focal_length,  720/2],
#   [0.,    0.,     1.]])

webcam_intrinsics = np.array([
  [1.40364870e+03, 0.00000000e+00, 351],
  [0.00000000e+00, 1.26593963e+03, 100],
  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

webcam_intrinsics_pandora = np.array([
  [1280, 0.00000000e+00, 660],
  [0.00000000e+00, 720, 280],
  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

webcam_intrinsics_dmd = np.array([
  [924.4637451171875, 0.0, 180],
  [0.0, 923.7325439453125, 362],
  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

trans_webcam_to_eon_front = np.dot(eon_dcam_intrinsics, np.linalg.inv(webcam_intrinsics))
trans_webcam_to_eon_front_pandora = np.dot(eon_dcam_intrinsics, np.linalg.inv(webcam_intrinsics_pandora))
trans_webcam_to_eon_front_dmd = np.dot(eon_dcam_intrinsics, np.linalg.inv(webcam_intrinsics_dmd))

trans_webcam_to_eon_front_1 = np.array([[1.42070485, 0.0, -30.16740088],
         [0.0, 1.42070485, 91.030837],
         [0.0, 0.0, 1.0]])

WEBCAM_WIDTH = 1152
WEBCAM_HEIGHT = 864


if __name__ == '__main__':
  print(trans_webcam_to_eon_front)