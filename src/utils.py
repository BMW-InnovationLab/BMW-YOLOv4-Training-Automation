import cv2
import os
import numpy as np


class Rectangle:
    def __init__(self, topleft, size):
        self.topleft = topleft
        self.size = size

    @classmethod
    def from_center(cls, center, size):
        return Rectangle(
            (center[0] - size[0] / 2, center[1] - size[1] / 2),
            size
        )

    def to_int(self):
        self.topleft = (int(self.topleft[0]), int(self.topleft[1]))
        self.size = (int(self.size[0]), int(self.size[1]))

    def get_topleft(self):
        return (self.topleft[0], self.topleft[1])

    def get_bottomright(self):
        return (self.topleft[0] + self.size[0], self.topleft[1] + self.size[1])

    def get_center(self):
        return (self.topleft[1] + self.size[1] / 2, self.topleft[0] + self.size[0] / 2)

    def get_left(self):
        return self.topleft[0]

    def get_right(self):
        return self.topleft[0] + self.size[0]

    def get_top(self):
        return self.topleft[1]

    def get_bottom(self):
        return self.topleft[1] + self.size[1]

    def get_area(self):
        return self.size[0] * self.size[1]

    @classmethod
    def calculate_iou(cls, r1, r2):
        left = max(r1.get_left(), r2.get_left())
        right = min(r1.get_right(), r2.get_right())
        bottom = min(r1.get_bottom(), r2.get_bottom())
        top = max(r1.get_top(), r2.get_top())

        return (right - left) * (bottom - top)



def get_capture_size(capture):
    return (int(capture.get(3)), int(capture.get(4)))

def get_output(filename, capture=None, capture_size=None, is_grey=False):
    path = 'media/output/{}.mp4'.format(filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if capture_size is None:
        capture_size = get_capture_size(capture)

    return cv2.VideoWriter(path, fourcc, 30.0, capture_size, not is_grey)

def get_sequence_length(path):
    return count_dir(path)

def get_frame_count(cap):
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def get_fps(cap):
    return int(cap.get(cv2.CAP_PROP_FPS))

# Vis-Drone
def get_vis_drone_path(sequence):
    vis_drone_path = os.environ['VIS_DRONE_PATH']
    return vis_drone_path + '/sequences/{}'.format(sequence)

def get_vis_drone_capture(sequence):
    path = get_vis_drone_path(sequence)
    return cv2.VideoCapture(path + '/%7d.jpg'), count_dir(path)

# KITTI
def get_kitti_path(sequence):
    kitti_path = os.environ['KITTI_PATH']
    img_path = '{}/data_odometry_gray/dataset/sequences/{}/image_0'.format(kitti_path, sequence)
    # pose_path = '{}/data_odometry_poses/dataset/poses/00.txt'.format(kitti_path)
    return img_path

def get_kitti_capture(sequence):
    path = get_kitti_path(sequence)
    return cv2.VideoCapture(path + '/%6d.png'), count_dir(path)

# Cenek Albl et al.
def get_cenek_path(sequence, camera):
    cenek_path = os.environ['CENEK_PATH']
    img_path = f'{cenek_path}/{sequence}/{camera}.mp4'
    ann_path = f'{cenek_path}/{sequence}/detections/{camera}.txt'
    return img_path, ann_path

def get_cenek_capture(sequence, camera):
    cap = cv2.VideoCapture(get_cenek_path(sequence, camera)[0])
    return cap, get_frame_count(cap)

def get_cenek_annotation(sequence, camera):
    return get_cenek_path(sequence, camera)[1]

def get_train_capture():
    path = 'media/train.mp4'
    return cv2.VideoCapture(path), 1e4

def count_dir(path):
    return len(os.listdir(path))

# Math utils
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return False, False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def line_angle(diff1, diff2):
    return np.arccos(np.dot(diff1, diff2) / (np.linalg.norm(diff1) * np.linalg.norm(diff2)))

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.

    Source: https://stackoverflow.com/a/16858283
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def read_flow(filename: str) -> np.ndarray:
    """Read flow field from a file

    Args:
        filename (str): path to the .flo file

    Returns:
        np.ndarray: (h, w, 2) array of the flow field
    """
    TAG_FLOAT = 202021.25

    with open(filename, 'rb') as f:
        flo_number = np.fromfile(f, np.float32, count=1)[0]
        assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number

        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

        return np.resize(data, (int(h), int(w), 2))