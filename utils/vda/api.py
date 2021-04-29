import cv2
from .dataio import combine_mask, getMaxOfMinRect, merge_strokes, preprocess_strokes
from .transform import perpective_transform, affine_transform
# from utils import sort_clock_wise
import os
import shutil
import numpy as np
import random
import math
import time
import multiprocessing as mp
def test_combine_mask():
    im_path = '/mnt/d/uet_vnu/thesis/dataset/augmentv3/content/debug/ETL8G_153881_0_input.png'
    mask_path = '/mnt/d/uet_vnu/thesis/dataset/augmentv3/content/debug/ETL8G_153881_label.npy'
    strokes = combine_mask(im_path, mask_path)
    [cv2.imwrite(f'debug_output/stroke_{i}.png', s) for i, s in enumerate(strokes)]
def batch_rand(*args):
    out = []
    for arg in args:
        x = random.random() * (arg[1] - arg[0]) + arg[0]
        out.append(x)
    return out
def get_coor(x, y, theta, center_x, center_y):
    theta = theta / 180 * math.pi
    rotatedX = x*math.cos(theta) - y*math.sin(theta)
    rotatedY = x*math.sin(theta) + y*math.cos(theta)

    # translate back
    x = rotatedX + center_x
    y = rotatedY + center_y
    return x, y
def iou(x, y):
    x = x > 0
    y = y > 0
    return (np.sum(x & y) / np.sum(x | y))
def rand_aff(strokes, contours):
    for i, s in enumerate(strokes):
        stroke_box = contours[i][1]
        center_range = [-0.05, 0.05]
        if stroke_box[-1] % 90 == 0:
            angle_range = [-5, 5]    
        else:
            angle_range = [-10, 10]
        scale_range = [0.9, 1.1]
        shift_range = [-1,1]
        c_x, c_y, angle, scale, d_x, d_y = batch_rand(center_range, center_range, angle_range, scale_range, shift_range, shift_range)
        
        c_y = (1 - c_y) * stroke_box[1][1] / 2
        c_x = (c_x - 1) * stroke_box[1][0] / 2
        c_x, c_y = get_coor(c_x, c_y, stroke_box[-1], stroke_box[0][0], stroke_box[0][1])
        strokes[i] = affine_transform(s, (c_x, c_y), angle, scale, (d_x, d_y))
    final = merge_strokes(strokes)
    final = 255 - final
    return final
def augment_one_image(p, root_dir, out_dir):
    p = os.path.split(p)[-1].split('.')[0]
    im_path = os.path.join(root_dir, f'{p}_0_input.png')
    mask_path = os.path.join(root_dir, f'{p}_label.npy')
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    strokes, masks = combine_mask(im_path, mask_path)
    strokes,contours = preprocess_strokes(strokes)
    return rand_aff(strokes, contours)
class VDA():
    def __init__(self):
        self.center_range = [-0.05, 0.05]
        self.angle_range = [-5, 5]    
        self.scale_range = [0.9, 1.1]        
        self.shift_range = [-1,1]
        self.root_dir = '/content/debug/content/debug/'
    def rand_aff(self, strokes, contours):
        for i, s in enumerate(strokes):
            stroke_box = contours[i][1]
            # # center_range = [-0.05, 0.05]
            # if stroke_box[-1] % 90 == 0:
            #     angle_range = [-5, 5]    
            # else:
            #     angle_range = [-10, 10]
            # scale_range = [0.9, 1.1]
            # shift_range = [-1,1]
            c_x, c_y, angle, scale, d_x, d_y = batch_rand(self.center_range, self.center_range, self.angle_range, self.scale_range, self.shift_range, self.shift_range)
            
            c_y = (1 - c_y) * stroke_box[1][1] / 2
            c_x = (c_x - 1) * stroke_box[1][0] / 2
            c_x, c_y = get_coor(c_x, c_y, stroke_box[-1], stroke_box[0][0], stroke_box[0][1])
            strokes[i] = affine_transform(s, (c_x, c_y), angle, scale, (d_x, d_y))
        final = merge_strokes(strokes)
        final = 255 - final
        return final
    def augment_one_image(self, p):
        p = os.path.split(p)[-1].split('.')[0]
        im_path = os.path.join(self.root_dir, f'{p}_0_input.png')
        mask_path = os.path.join(self.root_dir, f'{p}_label.npy')
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        strokes, masks = combine_mask(im_path, mask_path)
        strokes,contours = preprocess_strokes(strokes)
        return self.rand_aff(strokes, contours)
    def update_config(self, cfg):
        cfg.center_range = self.center_range
        cfg.angle_range = self.angle_range
        cfg.scale_range = self.scale_range
        cfg.shift_range = self.shift_range