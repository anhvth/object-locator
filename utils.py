import cv2
import numpy as np
import tensorflow as tf

def create_targets_from_points(points, height, width):
    masks = []
    zero_mask = np.zeros([height, width], dtype=np.float32)
    for i, point in enumerate(points):
        for x, y in point:
            zero_mask[y, x] = 1
        masks.append(zero_mask)
    masks = np.array(masks)
    return masks
        


def draw_points(imgs, locs, r=1, fuse=True):
    rv = []
    for img, loc in zip(imgs, locs):
        mask = np.zeros_like(img)
        for y, x in loc:
            cv2.circle(mask, (x, y), r, 255, -1)
            
        if fuse:
            rv.append(img*.5+mask*.5)
        else:
            rv.append(mask)
    return rv

