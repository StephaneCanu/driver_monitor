import cv2
import numpy as np
import os
import glob


# Define the dimensions of checkerboard
CHECKERBOARD = (6, 9)

# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []

# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored
# in a given directory. Since no path is
# specified, it will take current directory
# jpg files alone
images = glob.glob('chessboardimage/*.jpeg')

for filename in images:
	image = cv2.imread(filename)
	grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	# If desired number of corners are
	# found in the image then ret = true
	ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

	# If desired number of corners can be detected then, refine the pixel coordinates and display them on the images of checker board
	if ret:
		threedpoints.append(objectp3d)

		# Refining pixel coordinates
		# for given 2d points.
		corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
		twodpoints.append(corners2)

		# Draw and display the corners
		image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)

		cv2.imshow('img', image)
		cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = image.shape[:2]

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
	threedpoints, twodpoints, grayColor.shape[::-1], None, None)


# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)

#  Camera matrix:
# [[1.40364870e+03 0.00000000e+00 1.71818673e+02]
#  [0.00000000e+00 1.26593963e+03 1.88905273e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
#
#  Distortion coefficient:
# [[ 2.95280688e-01 -2.76477572e+02 -1.12579305e-02  4.38758831e-02
#    5.69382055e+03]]
#
#  Rotation Vectors:
# (array([[0.07918794],
#        [0.50973435],
#        [0.03263994]]), array([[ 0.25703561],
#        [ 0.40232497],
#        [-0.04004424]]), array([[-0.07276665],
#        [-0.52416911],
#        [ 1.4916419 ]]), array([[-0.30735717],
#        [-0.26326838],
#        [ 1.50383161]]), array([[-0.49153282],
#        [-0.39192004],
#        [ 1.45011617]]), array([[ 0.00108505],
#        [-0.39990645],
#        [ 0.15292718]]))
#
#  Translation Vectors:
# (array([[  2.34198675],
#        [ -5.90548776],
#        [110.74635936]]), array([[-11.05091994],
#        [-13.1011425 ],
#        [ 93.30616969]]), array([[ 11.4285394 ],
#        [-14.49392664],
#        [102.29848544]]), array([[ -2.57859213],
#        [ -5.4510143 ],
#        [110.72389876]]), array([[  3.34450507],
#        [-14.14839758],
#        [101.07101496]]), array([[-5.41737954e-02],
#        [-7.40325079e+00],
#        [ 1.11046984e+02]]))

