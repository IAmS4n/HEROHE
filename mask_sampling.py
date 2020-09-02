import cv2
import numpy as np
from numpy.random import choice


class SmallRegionError(Exception):
    pass



def region_counter(inp_c_sum, sx=1, sy=1):
    assert sx >= 1 and sy >= 1
    # pad inp :(befor_x, after_x) (befor_y, after_y)
    o = np.pad(inp_c_sum, ((0, sx), (0, sy)), "edge")
    dx = np.pad(inp_c_sum, ((sx, 0), (0, 0)), 'constant', constant_values=0)
    dx = np.pad(dx, ((0, 0), (0, sy)), "edge")

    dy = np.pad(inp_c_sum, ((0, 0), (sy, 0)), 'constant', constant_values=0)
    dy = np.pad(dy, ((0, sx), (0, 0)), "edge")

    dxy = np.pad(inp_c_sum, ((sx, 0), (sy, 0)), 'constant', constant_values=0)

    # out[x,y] == inp[x,y] + inp[x+1,y] + .. + inp[x+sx-1,y]
    # + inp[x,y+1] + ...
    # ...
    # + inp[x,sy-1] + ...

    return (o - dx - dy + dxy)[sx - 1:sx - 1 + inp_c_sum.shape[0], sy - 1:sy - 1 + inp_c_sum.shape[1]]

class MaskSingleBoxSampler:
    def __init__(self, mask, margin_x=0, margin_y=0, x_scale=1., y_scale=1.):
        assert type(mask) is np.ndarray and mask.dtype == bool
        self.mask = mask
        self.x_scale = x_scale
        self.y_scale = y_scale

        mask_as_image = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_as_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise SmallRegionError("Not contours detected")

        y1, x1, dy, dx = cv2.boundingRect(np.concatenate(contours))
        y2 = y1 + dy
        x2 = x1 + dx

        x1 = max(x1 - 1, margin_x)
        y1 = max(y1 - 1, margin_y)
        x2 = min(x2 + 1, mask.shape[0] - margin_x)
        y2 = min(y2 + 1, mask.shape[1] - margin_y)

        if x1 > x2 or y1 > y2:
            raise SmallRegionError("x1:%d x2:%d \t y1:%d y2:%d" % (x1, x2, y1, y2))

        self.x_range = (x1, x2)
        self.y_range = (y1, y2)

    def box(self):
        xbox = (round(self.x_range[0] * self.x_scale), round(self.x_range[1] * self.x_scale))
        ybox = (round(self.x_range[0] * self.y_scale), round(self.x_range[1] * self.y_scale))
        return xbox, ybox

    def sample(self):
        while True:
            # note : randint cause missing some points
            x = np.random.uniform(low=self.x_range[0], high=self.x_range[1])
            y = np.random.uniform(low=self.y_range[0], high=self.y_range[1])
            if self.mask[round(x), round(y)]:
                return round(x * self.x_scale), round(y * self.y_scale)
