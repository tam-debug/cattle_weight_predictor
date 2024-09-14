YOLO_MODEL_SIZES = ["n", "s", "m", "l", "x"]

K_FOLD_DATASET_INFO_FILENAME = "k_fold_data_split.csv"

TRAINING_RESULTS_HEADER = [
    "Timestamp",
    "Model Path",
    "Val Fold Mean IOU",
    "Val Fold StdDev IOU",
    "Test Fold Mean IOU",
    "Test Fold StdDev IOU",
]

# Tune
ITERATIONS = 300
HYPERPARAM_ARGS = {
    "epochs": [50, 150],
    "patience": 50,
    "batch": 50,
    "project": None,
    "name": None,
    "exist_ok": False,
    "optimizer": ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
    "single_cls": True,
    "cos_lr": [True, False],
    "lr0": [0.0001, 0.01],
    "lrf": [0.0001, 0.01],
    "momentum": [0.9, 1],
    "weight_decay": [0.0001, 0.001],
    "warmup_epochs": [1, 5],
    "warmup_momentum": [0.5, 1],
    "box": [1, 10],
    "cls": [0.5, 0.9],
}
DATA_AUGMENT_ARGS = {
    "hsv_h": [0, 1],
    "hsv_s": [0, 1],
    "hsv_v": [0, 1],
    "degrees": [0, 1],
    "translate": [0, 1],
    "scale": [0, 1],
    "shear": [-180, 180],
    "perspective": [0, 0.001],
    "flipud": [0, 1],
    "fliplr": [0, 1],
    "bgr": [0, 1],
    "mosaic": [0, 1],
    "mixup": [0, 1],
    "copy_paste": [0, 1],
    "auto_augment": ["randaugment", "autoaugment", "augmix"],
    "erasing": [0, 0.9],
    "crop_fraction": [0.1, 1],
}
TUNE_TRAIN_RESULTS_FILENAME = "tune_train_results.csv"
TUNE_TRAIN_RESULTS_HEADER = [
    "Timestamp",
    "With Head",
    "Model Path",
    "Val Mean IOU",
    "Val StdDev IOU",
]
TUNE_TEST_RESULTS_HEADER = [
    "Timestamp",
    "With Head",
    "Model Path",
    "Val Mean IOU",
    "Val StdDev IOU",
]
TUNE_TEST_RESULTS_FILENAME = "tune_test_results.csv"
BEST_MODEL_RESULTS_FILENAME = "best_model.csv"
BEST_MODEL_RESULTS_HEADER = [
    "Timestamp",
    "With Head",
    "Model_Size",
    "Model Index",
    "Val Mean IOU",
    "Val StdDevIOU",
    "Test Mean IOU",
    "Test StdDevIOU",
]


# Validate
CONFIDENCE_SCORE_THRESHOLD = 0.5

# TrainTestDataset builder
IMAGES_DIR = "images"
LABELS_DIR_WITH_HEAD = "labels/cattle_with_head"
LABELS_DIR_WITHOUT_HEAD = "labels/cattle_without_head"

# Weight model dataset
SCALE_DIMENSION = 640
DEPTH_MASK_DIMENSION = (640, 640)  # Height, Width

# Weight filenames
TRAIN_LOSS_PLOT_FILENAME = "train_loss_plot.png"
VAL_LOSS_PLOT_FILENAME = "val_loss_plot.png"
VAL_PREDICTIONS = "val_predictions.csv"
VAL_METRICS = "val_metrics.csv"

