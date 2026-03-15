import os

# Image settings
TARGET_H, TARGET_W = 360, 480
COORD_VECTOR_SIZE = 12

# Training settings
TRAIN_COUNT = 300
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4

# Base dataset paths (Kaggle)
BASE_INPUT_DIR = '/kaggle/input/datasets/relufrank/fs2020-runway-dataset'
BASE_LABEL_DIR = os.path.join(BASE_INPUT_DIR, 'labels')

IMAGES_TRAIN_DIR = os.path.join(BASE_INPUT_DIR, '1920x1080', '1920x1080', 'train')
MASKS_TRAIN_DIR = os.path.join(BASE_LABEL_DIR, 'labels', 'areas', 'train_labels_1920x1080')

IMAGES_TEST_DIR = os.path.join(BASE_INPUT_DIR, '1920x1080', '1920x1080', 'test')
MASKS_TEST_DIR = os.path.join(BASE_LABEL_DIR, 'labels', 'areas', 'test_labels_1920x1080')

COORDS_TRAIN_FILE = os.path.join(
    BASE_LABEL_DIR, 'labels', 'lines', 'train_labels_640x360.json'
)

COORDS_TEST_FILE = os.path.join(
    BASE_LABEL_DIR, 'labels', 'lines', 'test_labels_640x360.json'
)