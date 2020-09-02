import argparse
import hashlib
import multiprocessing
import random

import torch
from tqdm.auto import tqdm

import models
from utils import *

parser = argparse.ArgumentParser(prog='HEROHE')
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--mask_dir', type=str, required=False, default="./herohe_test_mask_cache/")

parser.add_argument('--make_mask', action='store_true')
parser.add_argument('--prevent_hash_in_mask', action='store_true')

parser.add_argument('--make_csv', action='store_true')
parser.add_argument('--csv_dir', type=str, required=False, default="./")

parser.add_argument('--patch_size', type=int, required=False, default=128)
parser.add_argument('--ensemble', type=int, required=False, default=64)
parser.add_argument('--patch_per_slide', type=int, required=False, default=256)
parser.add_argument('--model_fsize', type=int, required=False, default=64)
parser.add_argument('--pool_type', type=str, required=False, default="RGM1-16")
parser.add_argument('--drop_out_f', type=float, default=0.5)

parser.add_argument('--pre_train_model', type=str, required=False, default="modified_efficientnet-b0_64")
parser.add_argument('--pre_train_model_path', type=str, required=False, default="model.pth")

# parser.add_argument('--use_diverse_selection', action='store_true')
# parser.add_argument('--diverse_selection_patch_num_scale', type=int, required=False, default=8)

args = parser.parse_args()


def get_pathes(sample_name):
    slide_path = os.path.join(args.test_dir, sample_name)

    if args.prevent_hash_in_mask:
        hashed_name = sample_name
    else:
        hashed_name = hashlib.md5(bytes(slide_path, "utf8")).hexdigest() + "_" + sample_name

    mask_numpy_path = os.path.join(args.mask_dir, "%s_entropy_mask.npy" % hashed_name)

    return slide_path, mask_numpy_path, hashed_name


os.makedirs(args.mask_dir, exist_ok=True)

if args.make_mask:
    mask_work_list = []
    for sample_name in sorted(os.listdir(args.test_dir)):
        if not sample_name.endswith(".mrxs"):
            continue

        slide_path, mask_numpy_path, hashed_name = get_pathes(sample_name)

        if not os.path.exists(mask_numpy_path):
            mask_work_list.append((slide_path, hashed_name, args.mask_dir))

    if len(mask_work_list) == 0:
        print("All masks is ready!")
    else:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(make_masks, mask_work_list)

elif args.make_csv:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device', device)

    model = models.Model(args.pre_train_model, args.model_fsize,
                         heads=[1, ],
                         heads_pool=[args.pool_type, ],
                         drop_outs=[None if args.drop_out_f <= 0 else args.drop_out_f, ])
    model.load_state_dict(torch.load(args.pre_train_model_path))
    model = model.to(device)
    model.eval()

    # find size
    patch_size_bach_parta = args.patch_size
    patch_size_nm = pxsize2nm((patch_size_bach_parta, patch_size_bach_parta), BACH_MPP)
    print("Patch Size nm:", patch_size_nm)
    wsi_patch_size_px = [round(float(patch_size_nm[0]) / float(HEROHE_MPP.x)),
                         round(float(patch_size_nm[1]) / float(HEROHE_MPP.y))]
    print("Patch Size px:", wsi_patch_size_px)

    trans = make_transform_for_patches_list(wsi_patch_size_px=wsi_patch_size_px, output_size=model.image_size,
                                            mean=StaticsProperty.mean, std=StaticsProperty.std)
    results = []
    sample_name_list = [sample_name for sample_name in sorted(os.listdir(args.test_dir)) if
                        sample_name.endswith(".mrxs")]
    for sample_name in tqdm(sample_name_list):

        slide_path, mask_numpy_path, _ = get_pathes(sample_name)

        try:
            patch_gen = get_patch_gen(slide_path=slide_path, wsi_patch_size_px=wsi_patch_size_px,
                                      mask_path=mask_numpy_path)
        except:
            print("Error in %s, filled by random!" % sample_name)
            patch_gen = None
            final_prob = 0.5
            final_pred = random.choice([0, 1])
            results.append((sample_name.replace(".mrxs", ""), final_prob, final_pred))

        if patch_gen is not None:
            probs = []
            for ensemble_rep in tqdm(range(args.ensemble)):
                inp = []
                for _ in range(args.patch_per_slide):
                    inp.append(next(patch_gen))
                inp = trans(inp)[None, :, :, :, :]
                inp = torch.tensor(inp, device=device)

                with torch.no_grad():
                    logit = model(inp, head_num=0)[:, 0]
                    cur_probs = torch.sigmoid(logit)
                probs.append(cur_probs.item())
                assert type(probs[-1]) == float and 0. <= probs[-1] <= 1.

            # print(sample_name, probs)
            print(sample_name, np.mean(probs), np.std(probs))

            final_prob = float(np.mean(probs))
            final_pred = round(final_prob)
            results.append((sample_name.replace(".mrxs", ""), final_prob, final_pred))

    make_csv(result=results, path=os.path.join(args.csv_dir, "Piaz.csv"))

else:
    raise ValueError("Nothing to do!")
