import os
from configs import *

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from Trainer import Trainer

from detectron2.data.datasets import register_coco_instances, load_coco_json


setup_logger()


os.makedirs(OUTPUT_DIR, exist_ok=True)
register_coco_instances(
    "train",
    {},
    TRAIN_COCO_JSON_PATH,
    TRAIN_IMAGE_PATH,
)
load_coco_json(
    TRAIN_COCO_JSON_PATH,
    TRAIN_IMAGE_PATH,
    "train",
)

classes = MetadataCatalog.get("train").thing_classes


cfg = get_cfg()
cfg.OUTPUT_DIR = "{}/model/".format(BASE_DIR)

cfg.merge_from_file(model_zoo.get_config_file(PRETRAINED_MODEL_URL))
cfg.DATASETS.TRAIN = ("train",)
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
