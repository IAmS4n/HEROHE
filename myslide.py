import math
import random

import numpy as np
import openslide
from PIL import Image
from skimage.measure import shannon_entropy

import mask_sampling

Image.MAX_IMAGE_PIXELS = 933120000


class MaskIsEmpty(Exception):
    pass


class MySlide(openslide.OpenSlide):
    safe_margin = 16
    informative_threshold_default = 0.5

    def read_region(self, location, level, size):
        # For making compatible dimension between slide and numpy
        res = super().read_region(location=location, level=level, size=size)
        res = res.convert('RGB')
        res = res.transpose(Image.TRANSPOSE)
        return res

    def point_convert(self, point, dest_size):
        # point(or points) in slide dimension map to new dimension
        x_scale = float(dest_size[0]) / float(self.dimensions[0])
        y_scale = float(dest_size[1]) / float(self.dimensions[1])
        if type(point) is not list:
            return round(x_scale * point[0]), round(y_scale * point[1])
        else:
            return [(round(x_scale * p[0]), round(y_scale * p[1])) for p in point]

    def set_mpp(self, x, y):
        self.MPPX = float(x)
        self.MPPY = float(y)

    def convert_size_to_px(self, patch_size_px=None, patch_size_nm=None):
        if patch_size_px is not None:
            pass
        elif patch_size_nm is not None:
            try:
                patch_size_px = [round(float(patch_size_nm[0]) / float(self.MPPX)),
                                 round(float(patch_size_nm[1]) / float(self.MPPY))]
            except AttributeError:
                raise AttributeError("Resolution must be set using set_mpp")
        else:
            raise ValueError("Patch size is unknown")
        return patch_size_px

    def convert_center_to_corner(self, center_loc, patch_size_px):
        assert center_loc[0] - round(patch_size_px[0] / 2) >= 0 and center_loc[0] + round(patch_size_px[0] / 2) < \
               self.dimensions[0]
        assert center_loc[1] - round(patch_size_px[1] / 2) >= 0 and center_loc[1] + round(patch_size_px[1] / 2) < \
               self.dimensions[1]
        corner_loc = (center_loc[0] - round(patch_size_px[0] / 2.),
                      center_loc[1] - round(patch_size_px[1] / 2.))
        return corner_loc

    def random_pos_generator(self, whole_slide=True, patch_size_px=None, patch_size_nm=None):
        # whole_slide = True: whole of the slide
        # whole_slide = False: sample from mask

        wsi_patch_size_px = self.convert_size_to_px(patch_size_px, patch_size_nm)
        margin_x = round(wsi_patch_size_px[0] / 2) + self.safe_margin
        margin_y = round(wsi_patch_size_px[1] / 2) + self.safe_margin

        if whole_slide:
            while True:
                # Note : random int base on mask size cause missing some points
                x = random.randint(margin_x, self.dimensions[0] - margin_x)
                y = random.randint(margin_y, self.dimensions[1] - margin_y)
                yield x, y
        else:
            try:
                mask_sampler = self.mask_sampler
                assert self.mask_sampler_wsi_patch_size_px == wsi_patch_size_px
            except AttributeError:
                raise ValueError("set_mask_sampler before using this mode!")

            while True:
                x, y = mask_sampler.sample()
                yield x, y

    def get_patch(self, center_loc, patch_size_px=None, patch_size_nm=None):
        patch_size_px = self.convert_size_to_px(patch_size_px, patch_size_nm)
        corner_loc = self.convert_center_to_corner(center_loc, patch_size_px)
        res = self.read_region(corner_loc, level=0, size=patch_size_px)
        return res

    def make_entropy_map(self, level, size=None, scale=None, window_size=2):
        img_size = self.level_dimensions[level]

        if size is not None:
            assert img_size[0] * size[1] == img_size[1] * size[0]
        elif scale is not None:
            size = [int(img_size[0] / scale), int(img_size[1] / scale)]
        else:
            raise ValueError("Size or scale must be set")

        dx = float(img_size[0]) / float(size[0])
        dy = float(img_size[1]) / float(size[1])

        assert dx >= 8. and dy >= 8., "For meaningful entropy size must be more lower than level size(%d,%d)" % img_size

        img = self.read_region((0, 0), level, size=img_size)
        img = np.array(img)

        assert img.dtype == np.uint8
        assert img.shape[:2] == img_size
        assert img.shape[2] == 3

        res = np.zeros(size).astype(np.float)
        margin_size = float(window_size) / 2.

        for x in range(0, size[0]):
            for y in range(0, size[1]):
                xx1, xx2 = int((x - margin_size) * dx), int((x + margin_size) * dx) + 1
                yy1, yy2 = int((y - margin_size) * dy), int((y + margin_size) * dy) + 1
                xx1 = max(0, xx1)
                yy1 = max(0, yy1)
                xx2 = min(xx2, img_size[0])
                yy2 = min(yy2, img_size[1])
                for c in range(3):
                    res[x, y] += shannon_entropy(img[xx1:xx2, yy1:yy2, c]) / 3.
        return res

    def _load_informative_mask(self, path, cache):
        try:
            informative_mask = self.informative_mask
        except AttributeError:
            informative_mask = np.load(path)
            if cache:
                self.informative_mask = informative_mask
        assert informative_mask.dtype == np.bool
        return informative_mask

    def _load_informative_mask_sum(self, path, cache):
        try:
            informative_mask_sum = self.informative_mask_sum
        except AttributeError:
            informative_mask = self._load_informative_mask(path=path, cache=False)
            informative_mask_sum = np.cumsum(np.cumsum(informative_mask, 0), 1)
            if cache:
                self.informative_mask_sum = informative_mask_sum
        return informative_mask_sum

    def check_valid_patch(self, mask_path, center_loc, patch_size_px=None, patch_size_nm=None, cache=True,
                          threshold=informative_threshold_default):
        wsi_patch_size_px = self.convert_size_to_px(patch_size_px, patch_size_nm)
        wsi_corner_loc = self.convert_center_to_corner(center_loc, wsi_patch_size_px)

        informative_mask_sum = self._load_informative_mask_sum(path=mask_path, cache=cache)

        mask_corner, mask_size = self.point_convert([wsi_corner_loc, wsi_patch_size_px], informative_mask_sum.shape)
        mask_size = max(1, mask_size[0]), max(1, mask_size[1])

        # ?2 is not in patch
        x1, x2 = mask_corner[0], mask_corner[0] + mask_size[0]
        y1, y2 = mask_corner[1], mask_corner[1] + mask_size[1]

        region_sum = informative_mask_sum[x2 - 1, y2 - 1]
        if y1 == 0 and x1 == 0:
            pass
        elif x1 == 0:
            region_sum -= informative_mask_sum[x2 - 1, y1 - 1]
        elif y1 == 0:
            region_sum -= informative_mask_sum[x1 - 1, y2 - 1]
        else:
            region_sum -= informative_mask_sum[x2 - 1, y1 - 1]
            region_sum -= informative_mask_sum[x1 - 1, y2 - 1]
            region_sum += informative_mask_sum[x1 - 1, y1 - 1]

        return region_sum > (threshold * np.prod(mask_size))

    def get_mask(self, path, center_loc, patch_size_px=None, patch_size_nm=None, cache=True):
        wsi_patch_size_px = self.convert_size_to_px(patch_size_px, patch_size_nm)
        wsi_corner_loc = self.convert_center_to_corner(center_loc, wsi_patch_size_px)

        informative_mask = self._load_informative_mask(path=path, cache=cache)

        mask_corner, mask_size = self.point_convert([wsi_corner_loc, wsi_patch_size_px], informative_mask.shape)
        mask_size = max(1, mask_size[0]), max(1, mask_size[1])

        # x_scale = float(mask.shape[0]) / float(self.dimensions[0])
        # y_scale = float(mask.shape[1]) / float(self.dimensions[1])
        # corner = int(x_scale * wsi_corner_loc[0]), int(y_scale * wsi_corner_loc[1])
        # size = max(1, int(x_scale * wsi_patch_size_px[0])), max(1, int(y_scale * wsi_patch_size_px[1]))

        selected_region = informative_mask[mask_corner[0]:mask_corner[0] + mask_size[0],
                          mask_corner[1]:mask_corner[1] + mask_size[1]]
        return selected_region

    def set_mask_sampler(self, mask_path, mode="MaskSingleBoxSampler", patch_size_px=None, patch_size_nm=None,
                         threshold=informative_threshold_default):
        informative_mask_sum = self._load_informative_mask_sum(path=mask_path, cache=False)

        x_scale = float(self.dimensions[0]) / float(informative_mask_sum.shape[0])
        y_scale = float(self.dimensions[1]) / float(informative_mask_sum.shape[1])

        wsi_patch_size_px = self.convert_size_to_px(patch_size_px, patch_size_nm)

        margin_in_mask_x = math.ceil(((wsi_patch_size_px[0] / 2.) + float(self.safe_margin)) / x_scale)
        margin_in_mask_y = math.ceil(((wsi_patch_size_px[1] / 2.) + float(self.safe_margin)) / y_scale)

        patch_in_mask_x = math.ceil(wsi_patch_size_px[0] / x_scale)
        patch_in_mask_y = math.ceil(wsi_patch_size_px[1] / y_scale)

        cnt_per_patch = mask_sampling.region_counter(informative_mask_sum, sx=patch_in_mask_x, sy=patch_in_mask_y)

        half_patch_in_mask_x = math.ceil(wsi_patch_size_px[0] / (2. * x_scale))
        half_patch_in_mask_y = math.ceil(wsi_patch_size_px[1] / (2. * y_scale))
        centralize_cnt_per_patch = np.pad(cnt_per_patch,
                                          ((half_patch_in_mask_x, 0), (half_patch_in_mask_y, 0)),
                                          'constant', constant_values=0)[:cnt_per_patch.shape[0],
                                   :cnt_per_patch.shape[1]]

        if np.sum(centralize_cnt_per_patch) == 0:
            raise MaskIsEmpty

        # initial mask
        mask = centralize_cnt_per_patch > threshold * patch_in_mask_x * patch_in_mask_y
        self.mask_sampler = None
        while self.mask_sampler is None:
            try:
                if threshold > 0. and np.sum(mask) < 0.001 * np.prod(mask.shape):
                    raise mask_sampling.SmallRegionError

                if mode == "MaskSingleBoxSampler":
                    self.mask_sampler = mask_sampling.MaskSingleBoxSampler(mask,
                                                                           margin_x=margin_in_mask_x,
                                                                           margin_y=margin_in_mask_y,
                                                                           x_scale=x_scale, y_scale=y_scale)
                else:
                    raise ValueError("Invalid mode")
            except mask_sampling.SmallRegionError:
                if threshold == 0.:
                    raise MaskIsEmpty

                threshold /= 2.
                if threshold < 1e-2:
                    threshold = 0.
                print("Threshold decrease to %f" % threshold)

                mask = centralize_cnt_per_patch > threshold * patch_in_mask_x * patch_in_mask_y

        # for preventing error
        self.mask_sampler_wsi_patch_size_px = wsi_patch_size_px

    def _load_annotation_mask(self, path, palette, cache):
        try:
            annotation_mask = self.annotation_mask
        except AttributeError:
            img = Image.open(path)
            img = img.convert('P')
            img = img.transpose(Image.TRANSPOSE)
            tmp_palette = np.array(img.getpalette())
            assert len(tmp_palette) == 3 * 256

            color_map = {}
            for i in range(0, 256 * 3, 3):
                for cid, cv in enumerate(palette):
                    if all(tmp_palette[i:i + 3] == np.array(cv)):
                        color_map[int(i / 3)] = cid

            tmp_img = np.array(img)

            unknown_val = 255
            annotation_mask = np.ones_like(tmp_img, dtype=np.uint8) * unknown_val

            for old_val, new_val in color_map.items():
                annotation_mask[tmp_img == old_val] = new_val

            assert not (annotation_mask == unknown_val).any()

            if cache:
                self.annotation_mask = annotation_mask

                # warnings.warn("Mask size is %d MB" % (int(sys.getsizeof(annotation_mask) / (2 ** 20)),))

        assert annotation_mask.dtype == np.uint8
        return annotation_mask

    def get_annotation_from_img(self, path, palette, center_loc, patch_size_px=None, patch_size_nm=None,
                                cache=True):
        wsi_patch_size_px = self.convert_size_to_px(patch_size_px, patch_size_nm)
        wsi_corner_loc = self.convert_center_to_corner(center_loc, wsi_patch_size_px)

        annotation_mask = self._load_annotation_mask(path=path, palette=palette, cache=cache)
        mask_corner, mask_size = self.point_convert([wsi_corner_loc, wsi_patch_size_px], annotation_mask.shape)
        mask_size = max(1, mask_size[0]), max(1, mask_size[1])

        selected_region = annotation_mask[mask_corner[0]:mask_corner[0] + mask_size[0],
                          mask_corner[1]:mask_corner[1] + mask_size[1]]

        # x_scale = float(annotation_mask.shape[0]) / float(self.dimensions[0])
        # y_scale = float(annotation_mask.shape[1]) / float(self.dimensions[1])
        # corner = int(x_scale * wsi_corner_loc[0]), int(y_scale * wsi_corner_loc[1])
        # size = max(1, int(x_scale * wsi_patch_size_px[0])), max(1, int(y_scale * wsi_patch_size_px[1]))
        # selected_region = annotation_mask[corner[0]:corner[0] + size[0], corner[1]:corner[1] + size[1]]
        return selected_region
