import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class OMR_Scanner:
    def __init__(self, sol_path=None, vertical_markers=20, horizontal_markers=18):
        if sol_path is not None:
            self.img = cv2.imread(sol_path,0)
            print(type(self.img))
            self.pheight, self.pwidth = self.img.shape[0], self.img.shape[1]
            self.marker_cnt = 2 * (vertical_markers+horizontal_markers)

    def set_base_path(self, path):
        self.BASE_PATH = path
        if self.BASE_PATH[-1] != '/':
            self.BASE_PATH += '/'

    def load_sheet(self, img_path, vertical_markers=20, horizontal_markers=18):
        self.img = cv2.imread(img_path,0)
        self.pheight, self.pwidth = self.img.shape[0], self.img.shape[1]
        self.marker_cnt = 2 * (vertical_markers+horizontal_markers)

    def find_markers(self, min_area, max_area, threshold=190):
        self.ret, self.thresh = cv2.threshold(self.img, threshold, 255, 0)
        self.img2, self.contours, self.hierarchy = cv2.findContours(self.thresh, 1, 2)
        self.rect_centers = []

        for idx in range(len(self.contours)):
            contour = np.array(self.contours[idx])[:,0,:]
            area = cv2.contourArea(contour)
            if area > min_area and area < max_area:
                rect_center = np.average(contour, axis=0)
                if rect_center[1] < 0.15*self.pheight or rect_center[1] > 0.8*self.pheight or rect_center[0] < 0.15*self.pwidth or rect_center[0] > 0.85*self.pwidth:
                    self.rect_centers.append(rect_center)

        if len(self.rect_centers) != self.marker_cnt:
            self.is_correct = 'False'
        else:
            self.is_correct = ' True'
        self.rect_centers = np.array(self.rect_centers)
        print(len(self.rect_centers))

    def show_sheet(self):
        fig = plt.figure(figsize=[21,29.7], dpi=150)
        plt.imshow(self.img2)
        rect_centers = np.array(self.rect_centers)
        plt.scatter(rect_centers[:,0], rect_centers[:,1], color='white')
        return fig

    def sort_points(self):
        center_list = np.ndarray.tolist(self.rect_centers)
        left_pane = []
        right_pane = []
        top_pane = []
        bottom_pane = []

        for center in center_list:
            if center[0] < self.pwidth*0.1:
                left_pane.append(center)
            elif center[0] > self.pwidth*0.9:
                right_pane.append(center)

        for idx in left_pane:
            center_list.remove(idx)
        for idx in right_pane:
            center_list.remove(idx)

        for center in center_list:
            if center[1] < self.pheight*0.5:
                top_pane.append(center)
            elif center[1] > self.pheight*0.5:
                bottom_pane.append(center)

        for idx in top_pane:
            center_list.remove(idx)
        for idx in bottom_pane:
            center_list.remove(idx)
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

        self.horizontal = np.zeros([18,2,2])
        self.vertical = np.zeros([20,2,2])

        for idx in range(len(left_pane)):
            self.vertical[idx,0,:] = left_pane[idx,:]
            self.vertical[idx,1,:] = right_pane[idx,:]
        for idx in range(len(top_pane)):
            self.horizontal[idx,0,:] = top_pane[idx,:]
            self.horizontal[idx,1,:] = bottom_pane[idx,:]

        return self.vertical, self.horizontal

    def rotation_conversion(self, point, angle):
        conv_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        point_conv = np.dot(conv_mat, point)
        return point_conv

    def deg2rad(self, angle):
        return angle * np.pi / 180

    def slope(self, pointset):
        x_diff = pointset[1,0] - pointset[0,0]
        y_diff = pointset[1,1] - pointset[0,1]
        slp = y_diff / x_diff
        return slp

    def lattice(self):
        self.lattice_points = np.zeros([20,18,2])
        vertical_rot, horizontal_rot = np.zeros(self.vertical.shape), np.zeros(self.horizontal.shape)
        for idx in range(len(self.vertical[:,0,0])):
            vertical_rot[idx,0,:] = self.rotation_conversion(self.vertical[idx,0,:], self.deg2rad(-45))
            vertical_rot[idx,1,:] = self.rotation_conversion(self.vertical[idx,1,:], self.deg2rad(-45))
        for idx in range(len(self.horizontal[:,0,0])):
            horizontal_rot[idx,0,:] = self.rotation_conversion(self.horizontal[idx,0,:], self.deg2rad(-45))
            horizontal_rot[idx,1,:] = self.rotation_conversion(self.horizontal[idx,1,:], self.deg2rad(-45))

        for h in range(18):
            for v in range(20):
                A = np.array([[self.slope(vertical_rot[v,:,:]), -1], [self.slope(horizontal_rot[h,:,:]), -1]])
                # print('A: ', A)
                # print(A.shape)
                b = np.array([self.slope(vertical_rot[v,:,:])*vertical_rot[v,1,0] -vertical_rot[v,1,1], self.slope(horizontal_rot[h,:,:])*horizontal_rot[h,1,0] -horizontal_rot[h,1,1]])
                # print('b: ', b)
                # print(b.shape)
                # print(np.dot(np.linalg.inv(A), b).shape)
                # print(np.matmul(np.linalg.inv(A), b).shape)
                self.lattice_points[v,h,:] = np.dot(np.linalg.inv(A), b)
        for x in range(len(self.lattice_points[0,:,0])):
            for y in range(len(self.lattice_points[:,0,0])):
                self.lattice_points[y,x,:] = self.rotation_conversion(self.lattice_points[y,x,:], self.deg2rad(45))
                self.lattice_points = np.int32(self.lattice_points)
        return self.lattice_points

    def read_response(self, problem_cnt, threshold=200, answersheet=False, radius=10):
        # read id, first
        if answersheet == False:
            id_string = ""
            for y in range(10, 20):
                for x in range(0, 10):
                    if self.examine_mark(centerpoint=[self.lattice_points[y,x,0], self.lattice_points[y,x,1]],
                                        radius=radius, threshold=threshold):
                        id_string += str(x)
                        continue

        # read the responses
        answer = []
        answers = []
        for y in range(min(problem_cnt,20)):
            answer = []
            for x in range(10, 14):
                if self.examine_mark(centerpoint=[self.lattice_points[y,x,0], self.lattice_points[y,x,1]],
                                    radius=radius, threshold=threshold):
                    answer.append(x-9)
                    continue
            answers.append(answer)
        if answersheet == False:
            return id_string, answers
        else:
            return answers

    def examine_mark(self, centerpoint, radius, threshold):
        circle = self.pixel_circle(centerpoint=centerpoint, radius=radius)
        # print(circle)
        avg = 0
        for point in circle:
            avg += self.img[point[1], point[0]]
        avg /= len(circle)
        if avg <= threshold:
            return True
        else:
            return False

    def pixel_circle(self, centerpoint, radius):
        # print("centerpoint: ", centerpoint, "radius: ", radius)
        circle_pixels = []
        center = np.array(centerpoint)
        rect = np.zeros([2*radius+1,2*radius+1])
        x_min, x_max = centerpoint[0] - radius, centerpoint[0] + radius
        # print("x_min: ", x_min, "x_max: ", x_max)
        y_min, y_max = centerpoint[1] - radius, centerpoint[1] + radius
        # print("y_min: ", y_min, "y_max: ", y_max)

        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                point = [x, y]
                scalar_dist = np.sqrt(np.sum((center-np.array(point))**2))
                if scalar_dist <= radius:
                    circle_pixels.append(point)
        # print("pixel count: ", circle_pixels)
        return np.array(circle_pixels)

class AnswerSheet:
    def __init__(self, solution):
        self.solution = solution
        self.question_cnt = len(self.solution)
        print("Solutions: ", self.solution)

    def mark_score(self, answer):
        is_correct = []
        correct_cnt = 0
        if len(answer) != self.question_cnt:
            return -1
        for idx in range(self.question_cnt):
            if self.solution[idx] == answer[idx]:
                is_correct.append(True)
                correct_cnt += 1
            else:
                is_correct.append(False)
        return correct_cnt, is_correct
