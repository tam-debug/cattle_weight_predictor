from torchvision.transforms import v2

YOLO_MODEL_SIZES = ["n", "s", "m", "l", "x"]

# Default Train
TRAINING_RESULTS_NAME = "training_results.csv"
TRAINING_RESULTS_HEADER = [
    "Timestamp",
    "Model Size",
    "Mean IOU",
    "Standard Deviation IOU",
]
TRAINING_MODEL_PARAMS = {
    "single_cls": True,
    "epochs": 30,
    "batch": 100,
}

# Tune
ITERATIONS = 300
HYPERPARAM_ARGS = {
    "epochs": [1, 100],
    "patience": 100,
    "batch": 30,
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
    "cls": [0.1, 0.9],
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
    # "auto_augment": ["randaugment", "autoaugment", "augmix"]
    "erasing": [0, 0.9],
    "crop_fraction": [0.1, 1],
}
TUNE_RESULTS_PATH = "results/k_fold_tune_with_head.csv"
BEST_MODEL_RESULTS = "results/best_model.csv"
TUNE_RESULTS_HEADER = [
    "Timestamp",
    "With Head",
    "Model_Size",
    "Model Number",
    "Mean IOU",
    "Standard Deviation IOU",
]

# Validate
CONFIDENCE_SCORE_THRESHOLD = 0.5
PREDICTION_FILENAME = "predictions.json"

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

