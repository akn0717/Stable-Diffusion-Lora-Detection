import cv2
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

setup_logger()


BASE_DIR = "."
OUTPUT_DIR = "{}/output".format(BASE_DIR)
PRETRAINED_MODEL_URL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
os.makedirs(OUTPUT_DIR, exist_ok=True)


load_coco_json(
    "dataset/job_14-2023_11_18_12_13_26-coco 1.0/annotations/instances_default.json",
    "dataset/job_14-2023_11_18_12_13_26-coco 1.0/images",
    "segmentation_train",
)
classes = MetadataCatalog.get("segmentation_train").thing_classes

cfg = get_cfg()
cfg.OUTPUT_DIR = "{}/model/".format(BASE_DIR)

cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("data_detection_train",)


cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512


cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_0000999.pth"
)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
predictor = DefaultPredictor(cfg)


plane_metadata = MetadataCatalog.get("segmentation_train")
test_dataset_dicts = get_detection_data("test")
for idx, d in enumerate(test_dataset_dicts):
    im = cv2.imread(test_dataset_dicts[idx]["file_name"])
    # cv2.imshow('Image preview'.format(str(idx)),im)

    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    v = Visualizer(
        im[:, :, ::-1],
        metadata=plane_metadata,
        scale=0.25,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction preview".format(str(idx)), out.get_image())

    cv2.waitKey(0)

cv2.destroyAllWindows()
