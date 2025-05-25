import os

MODEL_TYPE = "A2C"
SHAPE = [20, 20]
STEPS_PER_SEC = 30
MONITOR_DIR = "tmp/monitor_log"
MODELS_DIR = "tmp/models"
MODEL_FILE_NAME = "SnakeModelv1"
EVAL_DIR = "tmp/eval"
RESULTS_DIR = "tmp/results"
TENSOR_LOG_DIR = "tmp/logs/tb"
USE_WALLS = True

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MONITOR_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TENSOR_LOG_DIR, exist_ok=True)
