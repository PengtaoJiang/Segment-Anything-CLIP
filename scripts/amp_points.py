# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import build_sam, SamPredictor 

import argparse
import json
import os
import numpy as np
from typing import Any, Dict, List

from tqdm import tqdm
import colorsys
import random
import clip
import torch
from clip_text import class_names_coco, class_names_ADE_150
from PIL import Image 
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--input-points-list",
    type=list,
    help=(
        "List of input points on the pre-segmentation mask."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    default="default",
    help="The type of model to load, in ['default', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def write_masks_to_folder(image: np.array, masks: List[Dict[str, Any]], path: str) -> None:
    # header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    # metadata = [header]
    colors = random_colors(3)
    # print(len(masks))
    mask_list = []
    for i, mask_data in enumerate(masks):
        mask = mask_data > 0.1
        
        masked_image = apply_mask(image, mask, colors[i])
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), masked_image)
        mask_list.append(mask)
    return mask_list


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def generate_text_embeddings(classnames, templates, model):
    with torch.no_grad():
        class_embeddings_list = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(args.device) #tokenize
            class_embedding = model.encode_text(texts) #embed with text encoder
            class_embeddings_list.append(class_embedding)
        class_embeddings = torch.stack(class_embeddings_list, dim=1).to(args.device)
    return class_embeddings

def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    # generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)
    predictor = SamPredictor(sam)

    ## clip
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    text_features = generate_text_embeddings(class_names_ADE_150, ['a clean origami {}.'], clip_model)#['a rendering of a weird {}.'], model)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in tqdm(targets):
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image_cp = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        h, w = image.shape[:2]
        ## input points 
        input_points_list = [[int(w*0.2), int(h*0.5)]]
        label_list = [1]

        points_cood = np.stack(input_points_list)
        points_label = np.array(label_list)
        masks, miou, _ = predictor.predict(point_coords=points_cood, point_labels=points_label)
        
        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        os.makedirs(save_base, exist_ok=True)
        mask_list = write_masks_to_folder(image_cp, masks, save_base)
        
        for i in range(len(mask_list)):
            mask = mask_list[i]
            image_new = image.copy()
            ind = np.where(mask > 0)
            image_new[mask == 0] = 0
            y1, x1, y2, x2 = min(ind[0]), min(ind[1]), max(ind[0]), max(ind[1])
            image_new = Image.fromarray(image_new[y1:y2+1, x1:x2+1])
            image_new = preprocess(image_new)
            
            image_features = clip_model.encode_image(image_new.unsqueeze(0).to(args.device))
            # Pick the top 5 most similar labels for the image
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze(0)
            similarity = (100.0 * image_features.float() @ text_features.float().T).softmax(dim=-1)
            values, indices = similarity[0].topk(5)

            print("\nTop predictions for i-th mask:\n")
            for value, index in zip(values, indices):
                print(f"{class_names_ADE_150[index]:>16s}: {100 * value.item():.2f}%")
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
