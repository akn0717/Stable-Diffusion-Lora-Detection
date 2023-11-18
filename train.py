import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from Trainer import Trainer

from detectron2.data.datasets import register_coco_instances, load_coco_json


setup_logger()


BASE_DIR = "."
OUTPUT_DIR = "{}/output".format(BASE_DIR)
PRETRAINED_MODEL_URL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


os.makedirs(OUTPUT_DIR, exist_ok=True)
register_coco_instances(
    "segmentation_train",
    {},
    "dataset/project_combined/annotations/instances_Train.json",
    "dataset/project_combined/images",
)
load_coco_json(
    "dataset/project_combined/annotations/instances_Train.json",
    "dataset/project_combined/images",
    "segmentation_train",
)

classes = MetadataCatalog.get("segmentation_train").thing_classes


cfg = get_cfg()
cfg.OUTPUT_DIR = "{}/model/".format(BASE_DIR)

cfg.merge_from_file(model_zoo.get_config_file(PRETRAINED_MODEL_URL))
cfg.DATASETS.TRAIN = ("segmentation_train",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(PRETRAINED_MODEL_URL)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 20000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
