from yacs.config import CfgNode as CN
from .models import MODELS
from .dataset import DATASETS
from .solver import SOLVER

_C = CN()

_C.MODEL = MODELS
_C.DATASETS = DATASETS
_C.SOLVER = SOLVER

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8

_C.OUTPUT_DIR = "outputs/default"

# ---------------------------------------------------------------------------- #
# Label Error Detection
# ---------------------------------------------------------------------------- #
_C.LABEL_ERROR = CN()

# Error Detection Data settings
_C.LABEL_ERROR.ERROR_DATA = CN()
_C.LABEL_ERROR.ERROR_DATA.TEST_DIR = "data/val/images"
_C.LABEL_ERROR.ERROR_DATA.TEST_COCO = "data/val/annotations_small.json"
_C.LABEL_ERROR.ERROR_DATA.TEST_METRICS = "data/dummy_metrics.csv"

# Analysis settings
_C.LABEL_ERROR.ANALYSIS = CN()
_C.LABEL_ERROR.ANALYSIS.THRESHOLD = 2.0
_C.LABEL_ERROR.ANALYSIS.OUTPUT_DIR = "outputs/label_error_analysis"
_C.LABEL_ERROR.ANALYSIS.N_CLUSTERS = 5
_C.LABEL_ERROR.ANALYSIS.LATENT_DIM = 256

# Visualization settings
_C.LABEL_ERROR.VIS = CN()
_C.LABEL_ERROR.VIS.SAVE_PLOTS = True
_C.LABEL_ERROR.VIS.DPI = 300
_C.LABEL_ERROR.VIS.PLOT_SIZE = CN()
_C.LABEL_ERROR.VIS.PLOT_SIZE.CLUSTER = [12, 8]
_C.LABEL_ERROR.VIS.PLOT_SIZE.METRICS = [15, 5]
_C.LABEL_ERROR.VIS.PLOT_SIZE.SUSPICIOUS = [10, 6]

# Create the global config object
cfg = _C.clone()