import numpy as np
import cv2
from scipy.spatial import distance
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