#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'demo'))
    print(os.getcwd())
except:
    pass

import posixpath
from io import BytesIO
from typing import Optional

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import tables as tb
from PIL import Image, ImageOps

from idf.reader import Reader
from idf.tables.bbox_det import InDetBBox
from idf.tables.rle import InRLE
from idf.writer import Writer
from inm.hdf import META
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

from tqdm import tqdm

#%%
# config_file = "../configs/inn_pcards_03_iou_freezebbn_template.yaml"
config_file = "../configs/inn_pcards_03_iou.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# MODEL_PTH = "../models/pcards-03-iou-template/model_final.pth"
MODEL_PTH = "../models/pcards-03-iou/model_final.pth"
cfg.merge_from_list(["MODEL.DEVICE", "cuda", "MODEL.WEIGHT", MODEL_PTH])

H5_synth = "/srv/datasets/pcards/val/pcards-synthetic-00-val-poly.h5"
H5_real = "/srv/datasets/pcards/test-real/pcards-real-00-test.h5"
H5 = H5_synth
INFERENCE_H5 = "pcards-synth-00-test-inference-iou.h5"
BASEPATH = os.path.dirname(H5)

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
pcards_demo = COCODemo(
    cfg,
    min_image_size=1278,
    confidence_threshold=0.7,
)

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
    for i_img, _ in enumerate(reader.image_uids):
        idf_img = reader.image_at(i_img)
        img_path = posixpath.join(basepath or "", idf_img.img_path)
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)[:, :, [2, 1, 0]]

        yield idf_img, img

#%%
reader = Reader.from_filepath(H5)
n_imgs = len(reader.image_uids)
with tb.open_file(INFERENCE_H5, "w") as hdf:
    writer = Writer(hdf, reader.info(META.DATASET))

    for idf_img, img in tqdm(idf_image(reader, BASEPATH), "Inference", total=n_imgs):
        predictions = pcards_demo.compute_prediction(img)
        writer.add_image(
            img_path=idf_img.img_path, img_id=idf_img.img_id,
            img_height=idf_img.img_height, img_width=idf_img.img_width
        )

        xywh_preds = predictions.convert("xywh")
        for bbox, score, label, mask in zip(
                xywh_preds.bbox, xywh_preds.get_field("scores"),
                xywh_preds.get_field("labels"),
                xywh_preds.get_field("mask")):
            bbox_ann = InDetBBox(
                label, bbox[1], bbox[0], bbox[3], bbox[2],
                bbox[3] * bbox[2], score
            )
            ann_id = writer.add_ann(idf_img.img_uid, bbox_ann)

            rle = mask_to_rle(mask.numpy())
            area = np.sum(rle[1::2])
            rle_ann = InRLE(label, area, rle)
            writer.append_ann(ann_id, idf_img.img_uid, rle_ann)

    writer.write_tables()


#%%
