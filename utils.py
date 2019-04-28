import cv2
import numpy as np

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
