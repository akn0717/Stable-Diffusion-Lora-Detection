from detectron2.engine import DefaultTrainer

from detectron2.data import (
    DatasetMapper,
    build_detection_train_loader,
)
import detectron2.data.transforms as T


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=[
                    # T.RandomBrightness(0.9, 1.1),
                    # T.RandomCrop("relative_range", (0.2,0.2)),
                    T.ResizeShortestEdge(
                        short_edge_length=[800, 1200],
                        max_size=1333,
                        sample_style="range",
                    ),
                    # T.RandomFlip(),
                ],
            ),
        )


#   @classmethod
#   def build_test_loader(cls, cfg, dataset_name = None):
#       return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, is_train=False, augmentations=[
#     #T.RandomApply(T.RandomCrop("relative_range", (0.2,0.2)),0.7),
#     T.ResizeShortestEdge(short_edge_length = [800,1000], max_size=1333, sample_style="choice"),
#  ]))

#   # refereces:https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
#   @classmethod
#   def build_evaluator(cls, cfg, dataset_name, output_folder = "./output".format(BASE_DIR)):
#       return COCOEvaluator(dataset_name=dataset_name, output_dir=output_folder)
