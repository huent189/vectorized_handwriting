import cv2
from .dataio import combine_mask, getMaxOfMinRect, merge_strokes, preprocess_strokes
from .transform import perpective_transform, affine_transform
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from .contour import modify_rect, rev_coordinate, get_coordinate, cal_min_rect_area, rect_distance
from .new_approach import SafeBox, compute_relationship, update_max_box
import random
MIN_AREA = 5

def get_non_border_cnt(im):
    h, w = im.shape
    contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_cnt = []
    for i,cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        if not(rect[0][0] < w * 0.1  or rect[0][0] > w * 0.9 or rect[0][1] < h * 0.1  or rect[0][1] > h * 0.9):
            text_cnt.append([cnt,rect])
        # return False
    return text_cnt

def preprocess_strokes(strokes):
    # remove border contour
    im_cnts = [get_non_border_cnt(s) for s in strokes]
    # clasify good and bad contour
    goods = []
    bads = []
    for i, s_cnts in enumerate(im_cnts):
        if len(s_cnts) == 1:
            area = cal_min_rect_area(s_cnts[0][1])
            if area < MIN_AREA:
                bads.append(s_cnts[0] + [i])
            else:
                goods.append(s_cnts[0] + [i])
        if len(s_cnts) > 1:
            max_area = 0
            max_cnt = None
            for cnt in s_cnts:
                area = cal_min_rect_area(cnt[1])
                if area >= max_area:
                    if max_cnt is not None:
                        bads.append(max_cnt + [i])
                    max_area = area
                    max_cnt = cnt
            if max_area < MIN_AREA:
                assert max_cnt is not None, f'{max_area}, {len(s_cnts)}'
                bads.append(max_cnt + [i])
            else:
                goods.append(max_cnt + [i])
    return goods
def rand_float_range(s, e):
    return random.random() * (e - s) + s
def sample_rotatedbox(safebox):
    org_rect = safebox.rect
    minbox = safebox.minbox
    maxbox = safebox.safebox
    w, h = org_rect[1]
    cx, cy = org_rect[0]
    theta = org_rect[-1]
    use_w = w > h
    src = [get_coordinate(w/2, h/2, cx, cy, theta), get_coordinate(w/2, -h/2, cx, cy, theta), 
                get_coordinate(-w/2, h/2, cx, cy, theta), get_coordinate(-w/2, -h/2, cx, cy, theta)]
    trg = []
    if use_w:
        l = -rand_float_range(0, safebox.safe_dl) - w /2
        r = rand_float_range(0, safebox.safe_dr) + w / 2
        if random.random() > 0.5:
            dh = rand_float_range(0, safebox.safe_dd)
        else:
            dh = -rand_float_range(0, safebox.safe_du)
        al = 0 if safebox.safe_dl < safebox.safe_dr else 1
        trg = [get_coordinate(r, h/2 + (1 - al) *dh, cx, cy, theta), get_coordinate(r, -h/2 + (1 - al) *dh, cx, cy, theta), 
                get_coordinate(l, h/2 + al *dh, cx, cy, theta), get_coordinate(l, -h/2 + al*dh, cx, cy, theta)]
    else:
        u = -rand_float_range(0, safebox.safe_du) - h /2
        d = rand_float_range(0, safebox.safe_dd) + h / 2
        if random.random() > 0.5:
            dw = rand_float_range(0, safebox.safe_dr)
        else:
            dw = -rand_float_range(0, safebox.safe_dl)
        au = 0 if safebox.safe_du < safebox.safe_dd else 1
        trg = [get_coordinate(w/2 + (1-au) * dw, d,cx, cy, theta), get_coordinate(w/2 + au*dw, u, cx, cy, theta), 
                get_coordinate(-w/2 + (1-au) * dw, d, cx, cy, theta), get_coordinate(-w/2 + au*dw,u, cx, cy, theta)]
    return src, trg
def tps_transform(im, srcshape, trgshape):
    srcshape = srcshape.reshape(1, -1, 2)
    trgshape = trgshape.reshape(1, -1, 2)
    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(srcshape.shape[1])]
    tps.estimateTransformation(trgshape, srcshape, matches)
    dst = tps.warpImage(im)
    return dst
def create_text_safebox(goods):
    good_boxes = [SafeBox(g[1]) for g in goods]
    for i in range(len(good_boxes)):
        for j in range(i+1, len(good_boxes)):
            update_max_box(good_boxes[i], good_boxes[j])
    return good_boxes
def random_tps(text_im, good_boxes):
    mul_stroke = [sample_rotatedbox(good_boxes[i]) for i in random.sample(range(len(good_boxes)), random.randint(1, len(good_boxes)))]
    src = np.array([s[0] for s in mul_stroke])
    trg = np.array([s[1] for s in mul_stroke])
    src.shape
    dst = tps_transform(text_im, trg, src)
    return dst

if __name__ == '__main__':
    import pickle
    import time
    s_time = time.time()
    im_path = '/mnt/d/uet_vnu/thesis/dataset/augmentv3/content/debug/ETL8G_004784_0_input.png'
    mask_path = '/mnt/d/uet_vnu/thesis/dataset/augmentv3/content/debug/ETL8G_004784_label.npy'
    text_im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    strokes, masks  = combine_mask(im_path, mask_path)
    goods_strokes = preprocess_strokes(strokes)
    good_boxes = create_text_safebox(goods_strokes)
    # with open('debug_output/ETL8G_004784_processedbox.png', 'wb') as f:
    #     pickle.dump(good_boxes, f)
    augmented_im = random_tps(text_im, good_boxes)
    e_time = time.time()
    print(e_time - s_time)
    cv2.imwrite('debug_output/augmented_im.png', augmented_im)
