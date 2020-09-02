import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_minimum
from torchvision import transforms

from my_transforms import ToNumpyAndNorm, ApplyOnPatches, StackPatches
from myslide import MySlide


def make_masks(slide_path, name, mask_result_dir):
    print("sample name : %s" % name)

    f_thresh = open(os.path.join(mask_result_dir, "%s_threshold.txt" % name), "w+")

    try:
        slide = MySlide(slide_path)
    except:
        print("Loading of WSI %s is failed!" % slide_path)

    thumb_level = 8
    thumb = np.array(slide.read_region((0, 0), level=thumb_level, size=slide.level_dimensions[thumb_level]))
    plt.imshow(thumb)
    plt.savefig(os.path.join(mask_result_dir, "%s_thumbnail.jpg" % name))
    plt.close()

    entropy_map = slide.make_entropy_map(level=4, scale=16)
    np.save(os.path.join(mask_result_dir, "%s_entropy_map.npy" % name), entropy_map)

    plt.imshow(np.clip(entropy_map * (256. / 10.), 0, 255).astype(np.int32))
    plt.savefig(os.path.join(mask_result_dir, "%s_entropy_map.jpg" % name))
    plt.close()

    plt.hist(entropy_map.flat, bins=32)
    plt.yscale('log')
    plt.savefig(os.path.join(mask_result_dir, "%s_entropy_hist.jpg" % name))
    plt.close()

    try:
        thresh = threshold_minimum(entropy_map[entropy_map > 3.0])
    except:
        thresh = threshold_minimum(entropy_map[entropy_map > 2.0])

    f_thresh.write("%s:%f\n" % (name, thresh))

    mask = entropy_map > thresh
    np.save(os.path.join(mask_result_dir, "%s_entropy_mask.npy" % name), mask)

    plt.imshow(mask.astype(np.int32) * 255)
    plt.savefig(os.path.join(mask_result_dir, "%s_entropy_mask.png" % name))
    plt.close()

    f_thresh.close()


def get_patch_gen(slide_path, wsi_patch_size_px, mask_path):
    threshold = 0.1

    slide = MySlide(slide_path)
    slide.set_mask_sampler(mask_path=mask_path, mode="MaskSingleBoxSampler", patch_size_px=wsi_patch_size_px,
                           threshold=threshold)
    cen_loc_gen = iter(slide.random_pos_generator(patch_size_px=wsi_patch_size_px, whole_slide=False))

    for center_loc in cen_loc_gen:
        yield slide.get_patch(center_loc=center_loc, patch_size_px=wsi_patch_size_px)


class BACH_MPP:
    x = 420.  # nm
    y = 420.  # nm


class HEROHE_MPP:
    x = 242.534722222222  # nm
    y = 242.64705882352899  # nm


class StaticsProperty:
    mean = [0.86949853, 0.71484338, 0.84091714]
    std = [0.16012984, 0.24309906, 0.1434607]


def pxsize2nm(input_px, mpp):
    return float(input_px[0]) * float(mpp.x), float(input_px[1]) * float(mpp.y)


def make_transform_for_patches_list(wsi_patch_size_px, output_size, mean, std):
    if output_size == wsi_patch_size_px:
        print("Same size")
        tr_per_patch = ToNumpyAndNorm(mean, std)
    else:
        tr_per_patch = transforms.Compose([transforms.Resize((output_size, output_size)), ToNumpyAndNorm(mean, std)])

    tr_main = transforms.Compose([
        ApplyOnPatches(tr_per_patch),
        StackPatches()
    ])

    return tr_main


def make_csv(result, path):
    with open(path, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(["caseID", "soft_prediction", "hard_prediction"])
        for caseID, soft_prediction, hard_prediction in result:
            filewriter.writerow([str(caseID), str(soft_prediction), str(hard_prediction)])
    print("CSV file:", os.path.abspath(path))
