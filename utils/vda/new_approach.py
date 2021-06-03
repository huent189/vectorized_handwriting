import numpy as np
import cv2
import math
from .contour import modify_rect, rev_coordinate, get_coordinate
# 0: no intersect
# 1: giao ơ giua
# 2: giao o biên
# 3: nam trong
# 4: nam ngoai
def compute_relationship(box1, box2):
    is_intersect, area = cv2.rotatedRectangleIntersection(box1, box2)
    if is_intersect == 0:
        return [0, 0]
    elif area.shape[0] <= 2:
            return [0, 0]
    elif is_intersect == 2:
        if box1[1][0] > box2[1][0]:
            return [4, 3]
        return [3, 4]
    else:
        rela = []
        ox, oy = np.mean(area, axis=(0,1))
        cx1, cy1 = box1[0]
        w1, h1 = box1[1]
        cx2, cy2 = box2[0]
        w2, h2 = box2[1]
        d1 = math.sqrt(((cx1 - ox)**2 + (cy1 - oy) ** 2) / (w1**2 + h1**2))
        if  d1 < 0.3:
            rela.append(2)
        else: 
            rela.append(1)
        d2 = math.sqrt(((cx2 - ox)**2 + (cy2 - oy) ** 2) / (w2**2 + h2**2))
        if d2 < 0.3:
            rela.append(2)
        else: 
            rela.append(1)
        return rela
class SafeBox():
    def __init__(self, rect):
        self.rect = rect
        self.minbox = modify_rect(self.rect, -self.rect[1][0] / 5, -self.rect[1][0] / 5, -self.rect[1][1] / 5, -self.rect[1][1] / 5)
        self.safe_dl = max(10, self.rect[1][0] / 4)
        self.safe_dr = max(10, self.rect[1][0] / 4)
        self.safe_du = max(10, self.rect[1][1] / 4)
        self.safe_dd = max(10, self.rect[1][1] / 4)
        self.safebox = modify_rect(self.rect, self.safe_dl, self.safe_dr, self.safe_du, self.safe_dd)
    def decrease_safebox(self, point):
        w, h = self.safebox[1]
        dx, dy = rev_coordinate(point[0], point[1], self.safebox[0][0], self.safebox[0][1], self.safebox[-1])
        l  = 0
        r = 0
        u = 0
        d = 0
        if dx < 0:
            l = - dx - w / 2
            if l + self.safe_dl <= 0:
                l = 0
            else:
                self.safe_dl += l
        else:
            r = dx - w / 2
            if r + self.safe_dr <= 0:
                r = 0
            else:
                self.safe_dr += r
        if dy < 0:
            u = -dy - h/2
            if u + self.safe_du <= 0:
                u = 0
            else:
                self.safe_du += u
        else:
            d = dy - h/2
            if d + self.safe_dd < 0:
                d = 0
            else:
                self.safe_dd += d
        self.safebox = modify_rect(self.safebox, l, r, u, d)
def update_max_box(sb1, sb2):
    old_rela = compute_relationship(sb1.rect, sb2.rect)
    new_rela = compute_relationship(sb1.safebox, sb2.safebox)
    if old_rela == new_rela:
        return
    else:
        i = 0
        while(i <= 10 and old_rela != new_rela):
            i += 1
            intersect, area = cv2.rotatedRectangleIntersection(sb1.safebox, sb2.safebox)
            point = np.mean(area, axis=(0, 1))
            sb1.decrease_safebox(point)
            sb2.decrease_safebox(point)
            new_rela = compute_relationship(sb1.safebox, sb2.safebox)