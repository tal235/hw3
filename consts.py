from pathlib import Path
import os

PROJECT_NAME = "HW3"

PROJECT_DIR = Path.home() / "hw3"
DATA_DIR = PROJECT_DIR / "data"

TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

CSV = '.csv'
TXT = '.txt'
YAML = '.yaml'

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

TEXT = "text"
INPUT_IDS = "input_ids"
LABEL = "label"

EPOCH = "epoch"
ITERATION = "iteration"
