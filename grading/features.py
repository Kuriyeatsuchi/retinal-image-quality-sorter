"""
Feature extraction for disease grading.

Each retina photo is cropped to the retina, padded to a square and passed
through ResNet18 pretrained on ImageNet. The 512-number output of the final
pooling layer is used as a description of the image; a simple classifier is
then trained on top of it (transfer learning with a frozen backbone).

The model is the ONNX Model Zoo's ResNet18 v1-7 (Apache-2.0), downloaded on
first use and verified by checksum. It is run with OpenCV's DNN module, so
no deep-learning framework needs to be installed.
"""

import hashlib
import os
import urllib.request

import cv2
import numpy as np

MODEL_URL = ("https://media.githubusercontent.com/media/onnx/models/main/validated/"
             "vision/classification/resnet/model/resnet18-v1-7.onnx")
MODEL_SHA256 = "4e8f8653e7a2222b3904cc3fe8e304cd8b339ce1d05fd24688162f86fb6df52c"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models",
                          "resnet18-v1-7.onnx")
FEATURE_LAYER = "flatten_170"   # output of the global average pool, 512 values

INPUT_SIZE = 320                # resolution fed to the network
CROP_SIZE = 384                 # resolution the retina crop is stored at
BACKGROUND_LEVEL = 15

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ensure_model(path=MODEL_PATH):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading ResNet18 weights to {path} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Checksum mismatch for {path}; delete it and try again.")
    return path


def crop_to_retina(image_bgr, size=CROP_SIZE):
    """Crop away the black background, pad to a square and resize."""
    mask = image_bgr.max(axis=2) > BACKGROUND_LEVEL
    ys, xs = np.where(mask)
    if len(ys) == 0:
        crop = image_bgr
    else:
        crop = image_bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side, 3), np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


class FeatureExtractor:
    def __init__(self, model_path=None):
        self.net = cv2.dnn.readNetFromONNX(ensure_model(model_path or MODEL_PATH))

    def _forward(self, rgb):
        x = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        self.net.setInput(x.transpose(2, 0, 1)[None].astype(np.float32))
        return self.net.forward(FEATURE_LAYER)[0]

    def __call__(self, image_bgr):
        """512 features for one image, averaged over the image and its mirror."""
        rgb = cv2.cvtColor(crop_to_retina(image_bgr), cv2.COLOR_BGR2RGB)
        return (self._forward(rgb) + self._forward(np.ascontiguousarray(rgb[:, ::-1]))) / 2

    def from_files(self, paths, verbose=True):
        features = []
        for i, path in enumerate(paths, 1):
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not read {path}")
            features.append(self(image))
            if verbose and (i % 100 == 0 or i == len(paths)):
                print(f"  features: {i}/{len(paths)}", flush=True)
        return np.array(features, dtype=np.float32)
