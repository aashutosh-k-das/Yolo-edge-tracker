"""INT8 entropy calibrator for TensorRT. Feeds representative images through the SAME
preprocessing used at inference time (deploy.utils.preprocess) - a mismatch here is the
most common cause of large INT8 accuracy drops."""
import glob
import os
import random

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from deploy.utils import preprocess


class YoloEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, imgsz=640, max_images=500, cache_file="calib.cache", seed=0):
        trt.IInt8EntropyCalibrator2.__init__(self)
        files = []
        for ext in ("jpg", "jpeg", "png", "bmp"):
            files += glob.glob(os.path.join(image_dir, "**", "*." + ext), recursive=True)
        files.sort()
        random.Random(seed).shuffle(files)
        self.files = files[:max_images]
        if not self.files and not os.path.exists(cache_file):
            raise FileNotFoundError("No calibration images found in " + image_dir)
        self.imgsz = imgsz
        self.batch_size = 1  # static-batch ONNX export -> calibration batch must be 1
        self.cache_file = cache_file
        self.index = 0
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * imgsz * imgsz * 4)
        print("INT8 calibration with {} images from {}".format(len(self.files), image_dir))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        while self.index < len(self.files):
            img = cv2.imread(self.files[self.index])
            self.index += 1
            if img is None:
                continue
            blob, _, _ = preprocess(img, self.imgsz, np.float32)
            cuda.memcpy_htod(self.device_input, np.ascontiguousarray(blob))
            if self.index % 50 == 0:
                print("  calibrated {}/{}".format(self.index, len(self.files)))
            return [int(self.device_input)]
        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print("Using existing calibration cache", self.cache_file)
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
