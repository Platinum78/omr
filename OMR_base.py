import numpy as np

def sort_points(centers, pwidth, pheight):
    center_list = np.ndarray.tolist(centers)
    left_pane = []
    right_pane = []
    top_pane = []
    bottom_pane = []

    for center in center_list:
        if center[0] < pwidth*0.1:
            left_pane.append(center)
        elif center[0] > pwidth*0.9:
            right_pane.append(center)

    for idx in left_pane:
        center_list.remove(idx)
    for idx in right_pane:
        center_list.remove(idx)

    for center in center_list:
        if center[1] < pheight*0.5:
            top_pane.append(center)
        elif center[1] > pheight*0.5:
            bottom_pane.append(center)

    for idx in top_pane:
        center_list.remove(idx)
    for idx in bottom_pane:
        center_list.remove(idx)
    print(len(left_pane) + len(right_pane) + len(bottom_pane) + len(top_pane))
    if len(left_pane) + len(right_pane) + len(bottom_pane) + len(top_pane) != 76:
        raise RuntimeWarning("Unsorted element left!")

    left_pane = np.array(left_pane)
    right_pane = np.array(right_pane)
    top_pane = np.array(top_pane)
    bottom_pane = np.array(bottom_pane)

    left_pane = left_pane[left_pane[:,1].argsort()]
    right_pane = right_pane[right_pane[:,1].argsort()]
    top_pane = top_pane[top_pane[:,0].argsort()]
    bottom_pane = bottom_pane[bottom_pane[:,0].argsort()]

    horizontal = np.zeros([18,2,2])
    vertical = np.zeros([20,2,2])

    for idx in range(len(left_pane)):
        vertical[idx,0,:] = left_pane[idx,:]
        vertical[idx,1,:] = right_pane[idx,:]
    for idx in range(len(top_pane)):
        horizontal[idx,0,:] = top_pane[idx,:]
        horizontal[idx,1,:] = bottom_pane[idx,:]

    return vertical, horizontal

def rotation_conversion(point, angle):
    conv_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    point_conv = np.dot(conv_mat, point)
    return point_conv

def deg2rad(angle):
    return angle * np.pi / 180

def slope(pointset):
    x_diff = pointset[1,0] - pointset[0,0]
    y_diff = pointset[1,1] - pointset[0,1]
    slp = y_diff / x_diff
    return slp

def mapping(min_orig, val_orig, max_orig, min_map, max_map):
    val_map = ((max_map - min_map) / (max_orig - min_orig)) * (val_orig - min_orig) + min_orig

def lattice(vertical, horizontal):
    lattice_points = np.zeros([20,18,2])
    vertical_rot, horizontal_rot = np.zeros(vertical.shape), np.zeros(horizontal.shape)
    for idx in range(len(vertical[:,0,0])):
        vertical_rot[idx,0,:] = rotation_conversion(vertical[idx,0,:], deg2rad(-45))
        vertical_rot[idx,1,:] = rotation_conversion(vertical[idx,1,:], deg2rad(-45))
    for idx in range(len(horizontal[:,0,0])):
        horizontal_rot[idx,0,:] = rotation_conversion(horizontal[idx,0,:], deg2rad(-45))
        horizontal_rot[idx,1,:] = rotation_conversion(horizontal[idx,1,:], deg2rad(-45))

    for h in range(18):
        for v in range(20):
            A = np.array([[slope(vertical_rot[v,:,:]), -1], [slope(horizontal_rot[h,:,:]), -1]])
            # print('A: ', A)
            # print(A.shape)
            b = np.array([slope(vertical_rot[v,:,:])*vertical_rot[v,1,0] -vertical_rot[v,1,1], slope(horizontal_rot[h,:,:])*horizontal_rot[h,1,0] -horizontal_rot[h,1,1]])
            # print('b: ', b)
            # print(b.shape)
            # print(np.dot(np.linalg.inv(A), b).shape)
            # print(np.matmul(np.linalg.inv(A), b).shape)
            lattice_points[v,h,:] = np.dot(np.linalg.inv(A), b)
    for x in range(len(lattice_points[0,:,0])):
        for y in range(len(lattice_points[:,0,0])):
            lattice_points[y,x,:] = rotation_conversion(lattice_points[y,x,:], deg2rad(45))
    return lattice_points

def lattice_linpol2d(vertical, horizontal):
    vertical_cnt = len(vertical[:,0,0])
    horizontal_cnt = len(horizontal[:,0,0])
    lattice_points = np.zeros([vertical_cnt, horizontal_cnt,2])
    for v in range(vertical_cnt):
        for h in range(horizontal_cnt):
            lattice_points[v,h,0] = horizontal[h,0,0] + (horizontal[h,1,0]-horizontal[h,0,0]) * ((h+2)/23)
            lattice_points[v,h,1] = vertical[v,0,1] + (vertical[v,1,1]-vertical[v,0,1]) * ((h+2)/23)
    return lattice_points

def read_value(lattice_points):

def examine_mark(centerpoint):
