#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
"""Eval models based on modified pycocotools."""

import argparse
from typing import NamedTuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ARGS(NamedTuple):
    dataset: str = "datasets/pcards/annotations/pcards-real-00-test.json"
    inference: str = "models/pcards-04-real_loss_weights_b/inference/coco_pcards_real_00_test/bbox.json"


args_real_reported = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test.json",
    "models/pcards-04-real_loss_weights_b/inference/coco_pcards_real_00_test/bbox.json",
)

# ---------------------------- M06 -------------------------------
args_M06_coco_pcards_synth_01_train_defects_only_class_type = ARGS(
    "datasets/pcards/annotations/pcards-synth-01-train-defects-only-class-type.json",
    "models/06-inn-pcards-04-real_labeled_loss_weights/inference/coco_pcards_synth_01_train_defects_only_class_type/bbox.json",
)
args_M06_coco_pcards_real_00_test_defects_only_class_type = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test-defects-only-class-type.json",
    "models/06-inn-pcards-04-real_labeled_loss_weights/inference/coco_pcards_real_00_test_defects_only_class_type/bbox.json",
)
args_M06_coco_pcards_real_00_test = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test.json",
    "models/06-inn-pcards-04-real_labeled_loss_weights/inference/coco_pcards_real_00_test/bbox.json",
)

# ---------------------------- M07 -------------------------------
args_M07_coco_pcards_real_00_train_defects_only_class_type = ARGS(
    "datasets/pcards/annotations/pcards-real-00-train-defects-only-class-type.json",
    "models/07-pcards-04-real_labeled_loss_weights/inference/coco_pcards_real_00_train_defects_only_class_type/bbox.json",
)
args_M07_coco_pcards_real_00_test_defects_only_class_type = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test-defects-only-class-type.json",
    "models/07-pcards-04-real_labeled_loss_weights/inference/coco_pcards_real_00_test_defects_only_class_type/bbox.json",
)
args_M07_coco_pcards_real_00_test = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test.json",
    "models/07-pcards-04-real_labeled_loss_weights/inference/coco_pcards_real_00_test/bbox.json",
)

# ---------------------------- M08 -------------------------------
args_M08_coco_pcards_synth_01_train_defects_only_class_type = ARGS(
    "datasets/pcards/annotations/pcards-synth-01-train-defects-only-class-type.json",
    "models/08-pcards-04-synth_labeled_loss_weights/inference/coco_pcards_synth_01_train_defects_only_class_type/bbox.json",
)
args_M08_coco_pcards_real_00_test_defects_only_class_type = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test-defects-only-class-type.json",
    "models/08-pcards-04-synth_labeled_loss_weights/inference/coco_pcards_real_00_test_defects_only_class_type/bbox.json",
)
args_M08_coco_pcards_real_00_test = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test.json",
    "models/08-pcards-04-synth_labeled_loss_weights/inference/coco_pcards_real_00_test/bbox.json",
)

# ---------------------------- M09 -------------------------------
args_M09_coco_pcards_synth_02_train_defects = ARGS(
    "datasets/pcards/annotations/train-type-1-b-synth-02.json",
    "models/09-pcards-04-synth_labeled_loss_weights/inference/coco_pcards_synthetic_02_filtered_train/bbox.json",
)
args_M09_coco_pcards_real_00_test_defects_only_class_type = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test-defects-only-class-type.json",
    "models/09-pcards-04-synth_labeled_loss_weights/inference/coco_pcards_real_00_test_defects_only_class_type/bbox.json",
)
args_M09_coco_pcards_real_00_test = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test.json",
    "models/09-pcards-04-synth_labeled_loss_weights/inference/coco_pcards_real_00_test/bbox.json",
)

# ----------------------------------------------------------------

args_real_00_test_labeled = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test-defects-labeled.json",
    "models/pcards-04-real_labeled_loss_weights/inference/coco_pcards_real_00_test_labeled/bbox.json",
)

args_real_00_train_labeled = ARGS(
    "datasets/pcards/annotations/pcards-real-00-train-defects-labeled.json",
    "models/pcards-04-real_labeled_loss_weights/inference/coco_pcards_real_00_train_labeled/bbox.json",
)

args_synth_01 = ARGS(
    "datasets/pcards/annotations/test-type-1-b-synth-01.json",
    "models/pcards-04-real_loss_weights_b/inference/coco_pcards_synthetic_01_test/bbox.json",
)

args_synth_01_ellipse = ARGS(
    "datasets/pcards/annotations/test-type-1-b-synth-01-ellipse.json",
    "models/pcards-04-real_loss_weights_b/inference/coco_pcards_synthetic_01_test-ellipse/bbox.json",
)

args_synth_01_line = ARGS(
    "datasets/pcards/annotations/test-type-1-b-synth-01-line.json",
    "models/pcards-04-real_loss_weights_b/inference/coco_pcards_synthetic_01_test-line/bbox.json",
)

args_synth_01_arc = ARGS(
    "datasets/pcards/annotations/test-type-1-b-synth-01-arc.json",
    "models/pcards-04-real_loss_weights_b/inference/coco_pcards_synthetic_01_test-arc/bbox.json",
)


args_synthetic = ARGS(
    "datasets/pcards/annotations/pcards-synthetic-00-val-poly-filtered.json",
    "models/pcards-04-real_loss_weights_b/inference/coco_pcards_synthetic_00_val/bbox.json",
)

args_iou = ARGS()

args_iou_template_synthetic = ARGS(
    "datasets/pcards/annotations/pcards-synthetic-00-val-poly-filtered.json",
    "models/pcards-03-iou-template/inference/coco_pcards_synthetic_00_val/bbox-filtered.json",
)

args_iou_template_real = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test.json",
    "models/pcards-03-iou-template/inference/coco_pcards_real_00_test/bbox.json",
)


args_iou_template_real_filtered = ARGS(
    "datasets/pcards/annotations/pcards-real-00-test-filtered.json",
    "models/pcards-03-iou-template/inference/coco_pcards_real_00_test/bbox-filtered.json",
)

args = args_M09_coco_pcards_real_00_test
dt = COCO(args.dataset)
res = dt.loadRes(args.inference)

iouThrs = np.linspace(0.15, 0.95, np.round((0.95 - 0.15) / 0.05) + 1, endpoint=True)
ceval = COCOeval(dt, res, "bbox", iouThrs=iouThrs)
ceval.params.useCats = 0
#%%
ceval.evaluate()
ceval.accumulate()
ceval.user_summarize()
