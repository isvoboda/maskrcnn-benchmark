# Maskrcnn benchmark in Innovatrics

## Training

Follow the main [README.md](../README.md).
Data in COCO format are available at `/mnt/nas.brno/image_tasks/1552_fingerpint_fore_back_segmentation/1586_sausage_fingerprints`.
The default backbone was deployed based on `configs/e2e_mask_rcnn_R_50_FPN_1x.yaml`



## Inference

Start `lab` in the docker container environment and copy the generated url including the token into your browser.
Start the lab service

~~~bash
cd docker
docker-compose run --rm --service-ports lab
~~~

In lab go to `demo` folder and launch the `slap-inference.ipynb`.
