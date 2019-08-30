#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filter-out annotations based on given area."""

import argparse
import json
from typing import NamedTuple
import numpy as np

import torch

from maskrcnn_benchmark.structures.bounding_box import BoxList

CHAMELEON = (1403, 938, 1963, 1233)
LMARK = (141, 1085, 378, 1257)
RMARK = (1826, 44, 2007, 279)

AREAS = [CHAMELEON, LMARK, RMARK]


class ARGS(NamedTuple):
    igt: str = "datasets/pcards/annotations/pcards-real-00-test.json"
    ogt: str = "datasets/pcards/annotations/pcards-real-00-test-filtered.json"
    idt: str = "models/pcards-03-iou-template/inference/coco_pcards_real_00_test/bbox.json"
    odt: str = "models/pcards-03-iou-template/inference/coco_pcards_real_00_test/bbox-filtered.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter Annotations"
    )
    parser.add_argument(
        "-igt",
        metavar="FILE",
        help="path to input dataset",
    )
    parser.add_argument(
        "-ogt",
        default="filtered_dtaset.json",
        metavar="FILE",
        help="path to output dataset",
    )
    parser.add_argument(
        "-idt",
        metavar="FILE",
        help="path to detections",
    )
    parser.add_argument(
        "-odt",
        metavar="FILE",
        help="path to detections",
    )

    return parser.parse_args()


def filter_bboxes(bboxes, area):
    condition_x = (bboxes[:, 0] < area[2]) & (bboxes[:, 2] > area[0])
    condition_y = (bboxes[:, 1] < area[3]) & (bboxes[:, 3] > area[1])
    indices = ~(condition_x & condition_y)

    return indices


def filter_predictions(predictions, area):
    filtered_predictions = []

    for pred in predictions:
        labels = pred.get_field("labels")
        scores = pred.get_field("scores")
        masks = pred.get_field("mask")

        ifiltered = filter_bboxes(pred.bbox, area)

        filtered = BoxList(pred.bbox[ifiltered], pred.size)
        filtered.add_field("labels", labels[ifiltered])
        filtered.add_field("scores", scores[ifiltered])
        filtered.add_field("mask", masks[ifiltered])

        filtered_predictions.append(filtered)

    return filtered_predictions


def filter_detections(detections, area):
    filtered_anns = []
    for ann in detections:
        bbox = np.asarray(ann["bbox"])
        bbox[2:] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        i_filtered = filter_bboxes(bbox.reshape([1, -1]), area)
        if i_filtered[0]:
            filtered_anns.append(ann)

    return filtered_anns


def filter_dataset(dataset, area):
    filtered_anns = []
    for ann in dataset["annotations"]:
        bbox = np.asarray(ann["bbox"])
        bbox[2:] = bbox[0] + bbox[2], bbox[1] + bbox[3]
        i_filtered = filter_bboxes(bbox.reshape([1, -1]), area)
        if i_filtered[0]:
            filtered_anns.append(ann)

    dataset["annotations"] = filtered_anns
    return dataset


def main():
    args = parse_args()

    # predictions = torch.load(args.predictions)
    with open(args.igt, "r") as fdgt, open(args.idt, "r") as fddt:
        gt = json.load(fdgt)
        dt = json.load(fddt)

    for i_area, area in enumerate(AREAS):
        print(f"Filter area {i_area}")
        gt = filter_dataset(gt, area)
        dt = filter_detections(dt, area)

    with open(args.ogt, "w") as gtf, open(args.odt, "w") as dtf:
        json.dump(gt, gtf)
        json.dump(dt, dtf)


if __name__ == "__main__":
    main()
