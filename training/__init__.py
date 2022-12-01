import os

TRAINING_DIR = os.path.dirname(__file__)
# Directory where logs (e.g. TensorBoard) will be placed
LOG_DIR = os.path.join(TRAINING_DIR, "logs")
TLDS_DIR = os.path.join(TRAINING_DIR, "tlds")
