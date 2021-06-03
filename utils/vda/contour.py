import numpy as np
import cv2
from scipy.spatial import distance
import math
def sort_clock_wise(pts):
    center = np.median(pts, axis=0)
    norm_pts = pts - center
    predicate = np.arctan2(norm_pts[:,0], norm_pts[:,1])
    order = np.argsort(predicate)
    return pts[order]
def cal_min_rect_area(bbox):
    area = bbox[1][1] * bbox[1][0]
    if area == 0:
        area = bbox[1][1] + bbox[1][0]
    return area  
def rect_distance(bbox1, bbox2):
    bbox1 = cv2.boxPoints(bbox1)
    bbox2 = cv2.boxPoints(bbox2)
    dis = distance.cdist(bbox1,bbox2)
    return dis.min()
def modify_rect(rect, l, r, u, d):
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[-1] / 180 * math.pi
    dw = (r - l) / 2
    dh = (d - u) / 2
    cx += dw*math.cos(theta) - dh*math.sin(theta)
    cy += dw*math.sin(theta) + dh*math.cos(theta)
    w += l + r
    h += u + d
    return tuple([[cx, cy],[w, h], rect[-1]])
def rev_coordinate(px, py, cx, cy, theta):
    # toa do dau vao theo anh oxy
    theta =  -(theta / 180 * math.pi)
    dx = px - cx
    dy = py - cy
    px = dx*math.cos(theta) - dy*math.sin(theta)
    py = dx*math.sin(theta) + dy*math.cos(theta)
    return px, py
def get_coordinate(px, py, cx, cy, theta):
    # toa do dau vao theo bouding box
    theta =  (theta / 180 * math.pi)
    dx = px*math.cos(theta) - py*math.sin(theta)
    dy = px*math.sin(theta) + py*math.cos(theta)
    px = dx + cx
    py = dy + cy
    return px, py