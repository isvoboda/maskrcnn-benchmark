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
from idf.writer import Writer
from inm.hdf import META
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

from tqdm import tqdm

#%%
config_file = "../configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.DEVICE", "cpu", "MODEL.WEIGHT", "../models/pcards/model_0010000.pth"])


#%%
pcards_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)


def idf_image(reader: Reader, basepath: Optional[str] = None):
    """Yields tuple of image id, and image."""
    for i_img, _ in enumerate(reader.image_uids):
        idf_img = reader.image_at(i_img)
        img_path = posixpath.join(basepath or "", idf_img.img_path)
        pil_img = Image.open(img_path).convert("RGB")
        img = np.array(pil_img)[:, :, [2, 1, 0]]

        yield idf_img, img


reader = Reader.from_filepath("../datasets/pcards/pcards-synthetic-00-test.h5")
with tb.open_file("../datasets/pcards/pcards-synthetic-00-test-inference.h5", "w") as hdf:
    writer = Writer(hdf, reader.info(META.DATASET))
    
    for idf_img, img in tqdm(idf_image(reader, "../datasets/pcards/test"), "Inference", total=1000):
        predictions = pcards_demo.compute_prediction(img)

        writer.add_image(
            img_path=idf_img.img_path, img_id=idf_img.img_id,
            img_height=idf_img.img_height, img_width=idf_img.img_width
        )

        xywh_preds = predictions.convert("xywh")
        for bbox, score, label in zip(
                xywh_preds.bbox, xywh_preds.get_field("scores"),
                xywh_preds.get_field("labels")):
            ann = InDetBBox(
                label, bbox[1], bbox[0], bbox[3], bbox[2],
                bbox[3]*bbox[2], score
            )
            writer.add_ann(idf_img.img_uid, ann)

    writer.write_tables()
