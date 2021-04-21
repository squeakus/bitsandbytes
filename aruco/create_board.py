import cv2
import cv2.aruco as aruco
import numpy as np

max_amount_of_markers_w = 3
max_amount_of_markers_h = 3
ar = aruco.DICT_6X6_1000
aruco_dict = aruco.Dictionary_get(ar)

# creat an aruco Board
grid_board = cv2.aruco.GridBoard_create(max_amount_of_markers_w, max_amount_of_markers_h, 0.05, 0.01, aruco_dict)

# convert to image
img = grid_board.draw((1000, 1000))

cv2.imwrite("out.png", img)

# # detected corners and ids
# corners, ids, rejected = aruco.detectMarkers(img, aruco_dict)

# # convert to X,Y,Z
# new_corners = np.zeros(shape=(len(corners), 4, 3))
# for cnt, corner in enumerate(corners):
#     new_corners[cnt, :, :-1] = corner

# # try to create a board via Board_create
# aruco.Board_create(new_corners, aruco_dict, ids)
