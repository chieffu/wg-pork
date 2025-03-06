# image_processor.py
import cv2
import numpy as np
import mss
import mss.tools
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from poker_cnn_classifier import PokerImageClassifier
from poker_cnn_classifier_3class import PokerImageClassifier3Class


class ImageProcessor:
    def __init__(self, regions):
        self.regions = regions
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.cnn = PokerImageClassifier()  # 每个子进程独立初始化 PokerImageClassifier
        self.cnn_3 = PokerImageClassifier3Class()

    def get_white_ratio(self, image, threshold=200):
        """检查图片中是否包含超过指定比例的白色像素"""
        gray = np.mean(image, axis=2)  # 更快地转换为灰度图
        white_pixels = np.count_nonzero(gray > threshold)
        total_pixels = gray.size
        return white_pixels / total_pixels

    def grab_screenshot(self, region_index):
        with mss.mss() as sct:
            screenshot = sct.grab(self.regions[region_index])
            frame = np.array(screenshot)
            white_ratio = self.get_white_ratio(frame)
            red_ratio = self._get_red_ratio(frame)
            image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return region_index, white_ratio, red_ratio, image

    def _get_red_ratio(self, image, lower_red1=(0, 100, 100), upper_red1=(10, 255, 255),
                       lower_red2=(160, 100, 100), upper_red2=(180, 255, 255)):
        """统计红色像素占比（HSV颜色空间）"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 定义红色在HSV中的两个范围（色相环首尾）
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        return np.count_nonzero(red_mask) / red_mask.size

    def detect_image_with_index(self, args):
        image, index = args
        predicted_class, confidence = self.cnn.detect_image(image)
        return index, predicted_class, confidence

    def detect_image_with_background(self, args):
        image, index = args
        predicted_class, confidence = self.cnn_3.detect_image(image)
        return index, predicted_class, confidence

    def process_images(self):
        futures = [self.executor.submit(self.grab_screenshot, i) for i in [0, 1]]
        results = [future.result() for future in futures]
        region_index1, white_ratio1, red_ratio1, image1 = results[0]
        region_index2, white_ratio2, red_ratio2, image2 = results[1]
        return white_ratio1,red_ratio1, image1, white_ratio2, red_ratio2,image2

    def detect_images(self, image1, image2):
        futures = [self.executor.submit(self.detect_image_with_index, (img, idx)) for idx, img in enumerate([image1, image2])]
        results = [future.result() for future in futures]
        index1, predicted_class1, confidence1 = results[0]
        index2, predicted_class2, confidence2 = results[1]
        return predicted_class1, confidence1, predicted_class2, confidence2

    def detect_images_background(self, image1, image2):
        futures = [self.executor.submit(self.detect_image_with_background, (img, idx)) for idx, img in
                   enumerate([image1, image2])]
        results = [future.result() for future in futures]
        index1, predicted_class1, confidence1 = results[0]
        index2, predicted_class2, confidence2 = results[1]
        return predicted_class1, confidence1, predicted_class2, confidence2
    def stop(self):
        self.executor.shutdown(wait=True)
