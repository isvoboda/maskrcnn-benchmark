#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os

try:
    os.chdir(os.path.join(os.getcwd(), "demo"))
    print(os.getcwd())
except:
    pass

import posixpath
from contextlib import ExitStack
from io import BytesIO
from typing import Optional

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from idf.reader import Reader
from idf.tables.image import RImage
from idf.tables.bbox_det import RDetBBoxV1
from idf.tables.rle import RRLE
from idf.writer import Writer
from idf.tables.paths import Paths
from PIL import Image, ImageOps
from tqdm import tqdm

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

#%%
# config_file = "../configs/inn_pcards_03_iou_freezebbn_template.yaml"
config_file = "../configs/pcards/09_inn_pcards_04_synth_defect_only.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# MODEL_PTH = "../models/pcards-03-iou-template/model_final.pth"
MODEL_PTH = "../models/09-pcards-04-synth_labeled_loss_weights/model_final.pth"
cfg.merge_from_list(["MODEL.DEVICE", "cuda", "MODEL.WEIGHT", MODEL_PTH])

H5_synth = "/srv/datasets/pcards/val/pcards-synthetic-00-val-poly.h5"
H5_real = (
    "/srv/datasets/pcards/test-real/pcards-real-00-test-defects-only-class-type.h5"
)
H5 = H5_real
INFERENCE_H5 = "inference-pcards-real-00-test-defects-only-class-type.h5"
BASEPATH = os.path.dirname(H5)

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
pcards_demo = COCODemo(cfg, min_image_size=1278, confidence_threshold=0.5,)


def mask_to_rle(binmask):
    mask_flat = np.ravel(binmask, order="F")
    starts = (np.flatnonzero(mask_flat[1:] != mask_flat[:-1]) + 1).tolist()
    if mask_flat[0] == 0:
        starts = [0] + starts + [len(mask_flat)]
    else:
        starts = [0, 0] + starts + [len(mask_flat)]
    rle_data = np.diff(starts)
    return rle_data


def idf_image(reader: Reader, basepath: Optional[str] = None):
    """Yields tuple of image id, and image."""
    for rimg in reader.images():
        img_path = posixpath.join(basepath or "", rimg.img_path)
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)[:, :, [2, 1, 0]]

        yield rimg, img


#%%
with ExitStack() as stack:
    h5s = {
        "read": stack.enter_context(tb.open_file(H5_real, "r")),
        "write": stack.enter_context(tb.open_file(INFERENCE_H5, "w")),
    }

    reader = Reader(h5s["read"])
    writer = Writer(h5s["write"], reader.metadata["dataset"])
    n_imgs = reader.table_of(Paths.images).nrows

    for rimg, img in tqdm(idf_image(reader, BASEPATH), "Inference", total=n_imgs):
        predictions = pcards_demo.compute_prediction(img)
        irimg = RImage(
            img_id=rimg.img_id,
            img_uid=rimg.img_uid,
            img_path=rimg.img_path,
            img_height=rimg.img_height,
            img_width=rimg.img_width,
            img_dataset=rimg.img_dataset,
        )
        writer.append_image(irimg)

        xywh_preds = predictions.convert("xywh")
        for bbox, score, label, mask in zip(
            xywh_preds.bbox,
            xywh_preds.get_field("scores"),
            xywh_preds.get_field("labels"),
            xywh_preds.get_field("mask"),
        ):
            det_bbox = RDetBBoxV1(
                img_uid=rimg.img_uid,
                cls_id=label,
                x=bbox[0],
                y=bbox[1],
                width=bbox[2],
                height=bbox[3],
                bbox_area=bbox[2] * bbox[3],
                det_score=score,
            )
            ann_id = writer.add_ann(det_bbox)

            rle = mask_to_rle(mask.numpy())
            area = np.sum(rle[1::2])
            rle_ann = RRLE(
                img_uid=rimg.img_uid,
                ann_id=ann_id,
                cls_id=label,
                rle_area=area,
                rle_data=rle,
                rle_img_width=rimg.img_width,
                rle_img_height=rimg.img_height,
            )
            writer.append_ann(rle_ann)

    writer.add_class_labels(
        reader.class_labels_of(Paths.coco_polys), [Paths.det_bboxes_v1, Paths.rles]
    )
    writer.write_tables()


#%%
