
import os
import sys
from pathlib import Path

import torch

from utils.torch_utils import model_info

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend

def run(weights=ROOT / 'pretrained/yolov5l.pt',  # model.pt path(s)
        data=ROOT / 'data/garbage.yaml',  # dataset.yaml path
        ):
    model = DetectMultiBackend(weights, device='cpu', data=data)
    # model_info(model)


if __name__ == "__main__":
    run()

