import torch
TORCH_VERSION = torch.__version__
CUDA_VERSION = torch.version.cuda

# basic setup
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os,son,cv2,random
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor,DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from roboflow import Roboflow
from detectron2.data.datasets import register_coco_instances

# download dataset from roboflow (with annotations)
ROBOTFLOW_API_KEY="YOUR_API_KEY_HERE"
rf=Roboflow(api_key=ROBOTFLOW_API_KEY)
project=rf.workspace().project("YOUR_PROJECT_NAME_HERE")
version=project.version(1)
dataset=version.download("coco-segmentation")
print(dataset.location)


train_json = os.path.join(dataset.location, "train/_annotations.coco.json")
train_images = os.path.join(dataset.location, "train")

test_json = os.path.join(dataset.location, "valid/_annotations.coco.json")  # si Roboflow usa "valid" en lugar de "test"
test_images = os.path.join(dataset.location, "valid")

register_coco_instances("my_dataset_train", {}, train_json, train_images)
register_coco_instances("my_dataset_test", {}, test_json, test_images)

# visualize some samples
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2_imshow(out.get_image()[:, :, ::-1])

# train custom Detectron2 detector
cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) # load cfg from moodel zoo
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

# look at the training curves in tensorboard
#%load_ext tensorboard
#%tensorboard --logdir output/

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("my_dataset_test", )
predictor = DefaultPredictor(cfg)

# inference with Detectron2 saved weights, visualize the label + confidence + bounding boxes on the test images
test_metadata = MetadataCatalog.get("my_dataset_test")
from detectron2.utils.visualizer import ColorMode
import glob
for imageNme in glob.glob("/content/Person_Detetction-1/test/*.jpg"):
    im = cv2.imread(imageNme)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=test_metadata,
                   scale=0.8
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])

# evaluation
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))

# we will see metrics like bbox mAP, segm mAP etc.

# save
f=open("detectron2_model_config.yaml","w")
f.write(cfg.dump())
f.close()