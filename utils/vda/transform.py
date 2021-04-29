import cv2
# from utils import sort_clock_wise
def perpective_transform(im, src_bbox, trg_bbox):
    # src_bbox = sort_clock_wise(src_bbox)
    # trg_bbox = sort_clock_wise(trg_bbox)
    M = cv2.getPerspectiveTransform(src_bbox, trg_bbox)
    warped = cv2.warpPerspective(im, M, (im.shape[1], im.shape[0]))
    return warped
def affine_transform(im, center, angle, scale, shift):
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += shift[0]
    M[1, 2] += shift[1]
    warped = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
    return warped