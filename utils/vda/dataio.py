import numpy as np
import cv2
from statistics import mode
from .contour import cal_min_rect_area, rect_distance
MIN_AREA = 5
MIN_IOU = 0.8
def combine_mask(im_path, mask_path):
    mask = np.load(mask_path)['arr_0']
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    assert len(im.shape) == 2, f'im shape was {im.shape}, only support gray image'
    stroke_ids, counts = np.unique(mask, return_counts=True)
    # print(mask['arr_0'])
    strokes = []
    stroke_masks = []
    for id in stroke_ids:
        if id == 0:
            continue
        stroke = np.zeros_like(im)
        stroke_pos = (mask[:,:,0] == id) | (mask[:,:,1] == id)
        stroke[stroke_pos] = im[stroke_pos]
        strokes.append(stroke)
        stroke_masks.append(stroke_pos)
    return strokes, stroke_masks
def remove_contours(im, cnt):
    mask = np.zeros_like(im)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    im[mask > 0] = 0
    return im
def get_non_border_cnt(im):
    h, w = im.shape
    contours, hierarchy = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    text_cnt = []
    for i,cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        if rect[0][0] < w * 0.1  or rect[0][0] > w * 0.9 or rect[0][1] < h * 0.1  or rect[0][1] > h * 0.9:
            tmp = remove_contours(im, cnt)
        else:
            text_cnt.append([cnt, rect])
        # return False
    return text_cnt
def merge_contour(src, trg, cnt):
    mask = np.zeros_like(src)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    trg[mask > 0] = src[mask > 0]
    src[mask > 0] = 0
    return trg, src
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
                    max_area = area
                    if max_cnt is not None:
                        bads.append(max_cnt + [i])
                    max_cnt = cnt
            if max_area < MIN_AREA:
                bads.append(max_cnt + [i])
            else:
                goods.append(max_cnt + [i])
    should_merge_after = []
    # print(len(im_cnts[5]))
    # print([b[-1] for b in bads])
    # print([b[-1] for b in goods])
    for i, b in enumerate(bads):
        merged = False
        for j, g in enumerate(goods):
            is_intersect, area = cv2.rotatedRectangleIntersection(b[1], g[1])
            # print(is_intersect, len(area) if area is not None else 0)
            if is_intersect == 2:
                merge_contour(strokes[b[-1]], strokes[g[-1]], b[0])
                merged = True
                continue
            elif is_intersect == 1 and len(area) > 4:
                merge_contour(strokes[b[-1]], strokes[g[-1]], b[0])
                merged = True
            elif is_intersect == 1 and len(area) == 4:
                area = np.array(area)
                sum_ax = area.mean(axis=(0,1))
                area = area[0,0] - sum_ax
                area = abs(area[0] * area[1] * 4)
                if area / cal_min_rect_area(b[1]) > MIN_IOU:
                    merge_contour(strokes[b[-1]], strokes[g[-1]], b[0])
                    merged = True
                    continue
            elif is_intersect == 1 and len(area) == 3:
                merge_contour(strokes[b[-1]], strokes[g[-1]], b[0])
                merged = True
                continue
            elif cal_min_rect_area(b[1]) < 5:
                remove_contours(strokes[b[-1]], b[0])
                merged = True
                continue
            elif g[-1] == b[-1]:
                if rect_distance(g[1], b[1]) < 5 or cal_min_rect_area(b[1]) / cal_min_rect_area(g[1]) > 0.7:
                    merged = True
                    should_merge_after.append([i, j])
                    continue
                
        # assert merged, b[-1]
   
    # update bouding box + contours
    for bg_pair in should_merge_after:
        new_cnt = np.concatenate([goods[j][0], bads[i][0]])
        goods[j][0] = new_cnt
        goods[j][1] = cv2.minAreaRect(new_cnt)
    good_idx = [b[-1] for b in goods]
    stroke_cnts = [b[:-1] for b in goods]
    return [strokes[i] for i in good_idx], stroke_cnts
def merge_strokes(strokes):
    merged_im = np.maximum.reduce(strokes)
    return merged_im
def getMaxOfMinRect(stroke):
    contours, hierarchy = cv2.findContours(stroke, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area = rect[1][0] * rect[1][1]
        if area > max_area:
            max_box = rect
            max_area = area
    # box is center, w, h, angle
    return max_box