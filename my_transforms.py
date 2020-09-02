import math
import random

import numpy as np
import scipy
from PIL import Image
from torchvision import transforms

DEFULTPADDING = ["reflect", "constant"][0]


class ImageRandomScaleSize:  # not tested!
    def __init__(self, scale_size_min, scale_size_max, scale_number, interpolation=Image.BILINEAR):
        self.scales = list(map(float, np.linspace(scale_size_min, scale_size_max, scale_number)))
        self.interpolation = interpolation

    def __call__(self, img):
        scale = random.choice(self.scales)
        new_size_w = round(img.size[0] * scale)
        new_size_h = round(img.size[1] * scale)
        new_size = (new_size_w, new_size_h)
        return img.resize(new_size, self.interpolation)


class RandomRandomCrop:
    def __init__(self, size, out_size=None, scale_size_diff=0., scale_number=10, pad_if_needed=True,
                 padding_mode=DEFULTPADDING):

        self.transforms = []
        for scale in map(float, np.linspace(1. - scale_size_diff, 1. + scale_size_diff, scale_number)):
            new_size = round(scale * float(size[0])), round(scale * float(size[1]))
            trans = transforms.RandomCrop(size=new_size, pad_if_needed=pad_if_needed, padding_mode=padding_mode)
            self.transforms.append(trans)

        if out_size is not None:
            self.final_resize = transforms.Resize(out_size)
        else:
            self.final_resize = None

    def __call__(self, img):
        rnd_crop = random.choice(self.transforms)
        rnd_img = rnd_crop(img)
        if self.final_resize is None:
            return rnd_img
        return self.final_resize(rnd_img)


class ValueAugmentOnFloatNp:
    def __init__(self, additive_gaussian=True, pblure=True, bh=True):
        self.additive_gaussian = additive_gaussian
        self.pblure = pblure
        self.bh = bh

    def __call__(self, x):
        # return x

        if self.additive_gaussian and random.random() < 0.5:
            x = self.additive_gaussian_noise(x)

        if self.pblure and random.random() < 0.5:
            x = self.partially_blur(x)

        if self.bh and random.random() < 0.5:
            x = self.black_hole(x)

        return x

    def additive_gaussian_noise(self, x):
        res = x + np.random.normal(scale=0.02, size=x.shape)
        return res.astype(np.float32)

    def random_soft_mask(self, x_shape, c=2.):
        x_max = float(x_shape[0])
        y_max = float(x_shape[1])

        # std:
        #     A  C
        #     C  B
        A = random.random() * x_max * c
        B = random.random() * y_max * c
        C = random.random() * max(0., math.sqrt(A * B) - 1.)  # for make sure semi positive
        mean = [random.random() * x_max, random.random() * y_max]
        cov = [[A, C], [C, B]]
        rv = scipy.stats.multivariate_normal(mean=mean, cov=cov)

        xEdges = np.linspace(0, x_shape[0], x_shape[0])
        yEdges = np.linspace(0, x_shape[1], x_shape[1])
        xMesh, yMesh = np.meshgrid(xEdges, yEdges)
        soft_mask = rv.pdf(np.stack((xMesh, yMesh), axis=2))
        soft_mask -= soft_mask.min()
        soft_mask /= soft_mask.max() + 1e-3
        return soft_mask.astype(np.float32)

    def black_hole(self, x):
        soft_mask = self.random_soft_mask(x.shape[1:], c=0.5)
        assert soft_mask.shape == x.shape[1:]

        x = np.multiply(x, 1. - soft_mask[None, :, :])
        x += np.multiply(np.zeros_like(x, dtype=np.float32), soft_mask[None, :, :])

        return x.astype(np.float32)

    def partially_blur(self, x):
        soft_mask = self.random_soft_mask(x.shape[1:], c=0.5)
        soft_mask_not = 1. - soft_mask

        blur_x = np.zeros_like(x, dtype=np.float32)
        assert x.shape[0] == 3
        for cn in range(x.shape[0]):
            blur_x[cn, :, :] = scipy.ndimage.filters.gaussian_filter(x[cn, :, :], sigma=2)

        x = np.multiply(x, soft_mask_not[None, :, :]) + np.multiply(blur_x, soft_mask[None, :, :])

        return x.astype(np.float32)


class RandomRotFlip:
    def __call__(self, pil_img):
        # there is 8 possible state

        rot = random.choice([None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
        if rot is not None:
            pil_img = pil_img.transpose(rot)

        # one of the flips is sufficient for reach all 8 possible

        if random.random() > .5:
            pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
            # PIL.Image.FLIP_LEFT_RIGHT
        return pil_img


class ApplyOnPatches:
    def __init__(self, f):
        self.f = f

    def __call__(self, patches):
        return [self.f(patch) for patch in patches]


class GeneratePatches:
    def __init__(self, f, num):
        self.f = f
        self.num = num

    def __call__(self, img):
        return [self.f(img) for _ in range(self.num)]


class StackPatches:
    def __call__(self, patches):
        try:
            res = np.stack(patches, axis=0)
        except:
            if type(patches) is list:
                for x in patches:
                    print(x.shape)
            elif type(patches) is np.ndarray:
                print(patches.shape)
            raise ValueError

        return res

        # return torch.stack(patches, dim=0)


class ToNumpyAndNorm:
    # https://github.com/pytorch/vision/blob/07cbb46aba8569f0fac95667d57421391e6d36e9/torchvision/transforms/functional.py#L192
    # https://github.com/pytorch/vision/blob/07cbb46aba8569f0fac95667d57421391e6d36e9/torchvision/transforms/functional.py#L43
    def __init__(self, mean, std, aug=None):
        assert len(std) == 3
        assert len(mean) == 3
        self.std = np.array(std).astype(np.float32)[:, None, None]
        self.mean = np.array(mean).astype(np.float32)[:, None, None]

        self.aug = aug

    def __call__(self, img):
        img = np.array(img)
        assert len(img.shape) == 3 and img.shape[-1] == 3, img.shape
        assert img.dtype == np.uint8
        # img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.
        img = np.rollaxis(img, 2).astype(np.float32) / 255.
        if self.aug is not None:
            img = self.aug(img)
        img = (img - self.mean) / (self.std + 1e-2)
        return img


class PrintShape:
    def shape_print(self, x, pref=""):
        if type(x) is list:
            print(pref, "list", len(x))
            for item in x:
                self.shape_print(item, pref=pref + "\t")
        elif hasattr(x, "shape"):
            print(pref, x.shape)
        elif hasattr(x, "size"):
            print(pref, x.size)
        else:
            raise NotImplementedError

    def __call__(self, x):
        self.shape_print(x)
        return x


class MakeBag:  # when resizing patch is efficient than resize whole of images
    def __init__(self, select_size, output_size, bag_size, augment_func, mean, std, scale_size_diff=0.,
                 padding_mode=DEFULTPADDING, debug=False):
        print("Making bag transform")

        additional_aug = None
        if augment_func is None:
            augment_func = lambda x: x
        # else:
        #     additional_aug = ValueAugmentOnFloatNp()

        if output_size == select_size and scale_size_diff == 0.:
            print("Same size")
            tr_per_patch = transforms.Compose([augment_func,
                                               ToNumpyAndNorm(mean, std, aug=additional_aug)])
        else:

            # in scale_size_diff mode is excepted!
            select_is_bigger = output_size < select_size
            print("select is bigger:", select_is_bigger)

            resize_to_output = transforms.Resize((output_size, output_size))
            tr_per_patch = transforms.Compose([resize_to_output if select_is_bigger else augment_func,
                                               augment_func if select_is_bigger else resize_to_output,
                                               ToNumpyAndNorm(mean, std, aug=additional_aug)])

        if scale_size_diff == 0.:
            rnd_patch_crop = transforms.RandomCrop((select_size, select_size), pad_if_needed=True,
                                                   padding_mode=padding_mode)
        else:
            rnd_patch_crop = RandomRandomCrop(size=(select_size, select_size),
                                              scale_size_diff=scale_size_diff,
                                              pad_if_needed=True,
                                              padding_mode=padding_mode)
        if debug:
            print("Debug Mode")
            print("Bag size:", bag_size)
            print("select_size:", select_size)
            print("output_size:", output_size)
            self.tr_main = transforms.Compose([
                GeneratePatches(rnd_patch_crop, bag_size),
                PrintShape(),
                ApplyOnPatches(tr_per_patch),
                PrintShape(),
                StackPatches(),
                PrintShape(),
            ])
        else:
            self.tr_main = transforms.Compose([
                GeneratePatches(rnd_patch_crop, bag_size),
                ApplyOnPatches(tr_per_patch),
                StackPatches()
            ])

    def __call__(self, img_pil):
        return self.tr_main(img_pil)


class ImageScaleSize:
    def __init__(self, scale_size, interpolation=Image.BILINEAR):
        self.scale_size = scale_size
        self.interpolation = interpolation

    def __call__(self, img):
        new_size_w = round(img.size[0] * self.scale_size)
        new_size_h = round(img.size[1] * self.scale_size)
        new_size = (new_size_w, new_size_h)
        return img.resize(new_size, self.interpolation)


class MakeBagBigPatch:  # when resizing whole of images is efficient than resize patch
    def __init__(self, patch_size, output_size, augment_func, mean, std, scale_size_diff=0., bag_size=1,
                 padding_mode=DEFULTPADDING):
        raise NotImplementedError

        if augment_func is None:
            augment_func = lambda x: x

        scale = float(output_size) / float(patch_size)
        print("Scale:", scale)

        if scale < 1:
            if scale_size_diff == 0.:
                rnd_patch_crop = transforms.RandomCrop((output_size, output_size), pad_if_needed=True,
                                                       padding_mode=padding_mode)
            else:
                # must be implement!
                self.tr_main = MakeBag(select_size=patch_size,
                                       output_size=output_size,
                                       bag_size=bag_size,
                                       augment_func=augment_func,
                                       mean=mean, std=std, scale_size_diff=scale_size_diff,
                                       padding_mode=padding_mode)
                return
                # rnd_patch_crop = RandomRandomCrop(size=(output_size, output_size),
                #                                   scale_size_diff=scale_size_diff,
                #                                   pad_if_needed=True,
                #                                   padding_mode=padding_mode)
            tr_per_patch = transforms.Compose([
                augment_func,
                ToNumpyAndNorm(mean, std)
            ])
            self.tr_main = transforms.Compose([
                ImageScaleSize(scale),
                GeneratePatches(rnd_patch_crop, bag_size),
                ApplyOnPatches(tr_per_patch),
                StackPatches()
            ])
        else:
            self.tr_main = MakeBag(select_size=patch_size,
                                   output_size=output_size,
                                   bag_size=bag_size,
                                   augment_func=augment_func,
                                   mean=mean, std=std, scale_size_diff=scale_size_diff,
                                   padding_mode=padding_mode)

            # rnd_patch_crop = transforms.RandomCrop((patch_size, patch_size), pad_if_needed=True,
            #                                        padding_mode=padding_mode)
            # tr_per_patch = transforms.Compose([
            #     augment_func,
            #     transforms.Resize((output_size, output_size)),
            #     transforms.ToTensor(),
            #     normalize_func
            # ])
            # self.tr_main = transforms.Compose([
            #     GeneratePatches(rnd_patch_crop, bag_size),
            #     ApplyOnPatches(tr_per_patch),
            #     StackPatches()
            # ])

    def __call__(self, img_pil):
        return self.tr_main(img_pil)
