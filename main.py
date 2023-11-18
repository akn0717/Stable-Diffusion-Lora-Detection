from tqdm import tqdm
import cv2
import os
import shutil

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger

setup_logger()


# Hyperparameters
PADDING = 20


os.makedirs("model", exist_ok=True)
cfg = get_cfg()
cfg.OUTPUT_DIR = "model/"


cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.DATASETS.TRAIN = ("data_detection_train",)


cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512


cfg.MODEL.WEIGHTS = os.path.join(
    cfg.OUTPUT_DIR, "model_final.pth"
)  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set a custom testing threshold
predictor = DefaultPredictor(cfg)

os.makedirs("output", exist_ok=True)
os.makedirs("output/original", exist_ok=True)

load_coco_json(
    "dataset/job_14-2023_11_18_12_13_26-coco 1.0/annotations/instances_default.json",
    "dataset/job_14-2023_11_18_12_13_26-coco 1.0/images",
    "segmentation_train",
)
classes = MetadataCatalog.get("segmentation_train").thing_classes

for c in classes:
    os.makedirs("output/" + c, exist_ok=True)

for idx, filename in enumerate(tqdm(os.listdir("input"))):
    filepath = "input/" + filename
    if (
        filepath.split(".")[-1] == "png"
        or filepath.split(".")[-1] == "webp"
        or filepath.split(".")[-1] == "jpg"
    ):
        im = cv2.imread(filepath)

        # copy current image to new folder
        shutil.copyfile(filepath, "output/original/" + filename)

        outputs = predictor(
            im
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

        original_height, original_width = im.shape[:2]
        for idx, bbox in enumerate(outputs["instances"].pred_boxes):
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = int(bbox[2])
            y1 = int(bbox[3])

            x0 = max(0, x0 - PADDING)
            y0 = max(0, y0 - PADDING)
            x1 = min(x1 + PADDING, original_width)
            y1 = min(y1 + PADDING, original_height)
            cropped_img = im[y0:y1, x0:x1, :]
            filename_no_ext = os.path.splitext(filename)[0]
            cv2.imwrite(
                "output/{}/{}_{}.png".format(
                    classes[outputs["instances"].pred_classes[idx]],
                    filename_no_ext,
                    str(idx),
                ),
                cropped_img,
            )
