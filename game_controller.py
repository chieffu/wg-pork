import asyncio
import socket
import cv2
import numpy as np
import pyautogui
import time
import os

from image_processor import ImageProcessor
from poker_cnn_classifier import PokerImageClassifier, Poker

os.environ['PYTHONIOENCODING'] = 'utf-8'
# 设置默认延迟为0.01秒
pyautogui.PAUSE = 0.003
# 设置最小持续时间为0.01秒
pyautogui.MINIMUM_DURATION = 0.002


class GameController:
    def __init__(self, x=1437, y=883, width=54, distance=146, hotkey_long='1', hotkey_hu='2', hotkey_he='3', log_callback=None, update_image_callback=None, show_hint_callback=None, websocket_server=None):
        self.x = x
        self.y = y
        self.width = width
        self.distance = distance
        self.hotkey_long = hotkey_long
        self.hotkey_hu = hotkey_hu
        self.hotkey_he = hotkey_he
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.regions = [
            (x, y, x + width, y + width),  # 龙牌截图区域
            (x + distance, y, x + distance + width, y + width),  # 虎牌截图区域
            (x - 121, y + 290, x - 121 + 100, y + 290 + 100),  # 龙下注点击区域
            (x + 253, y + 290, x + 253 + 100, y + 290 + 100),  # 虎下注点击区域
            (x + 45, y + 238, x + 45 + 100, y + 238 + 70)  # 和下注点击区域
        ]
        self.white_radios = [0.0, 0.0]
        self.is_paused = False
        self.is_running = True
        self.log_callback = log_callback
        self.update_image_callback = update_image_callback
        self.show_hint_callback = show_hint_callback
        self.imageProcessor = ImageProcessor(self.regions)

        # 状态变量
        self.has_seen_card_back = [False, False]
        self.first_card_back_time = [None, None]
        self.status = [0, 0]  # 0: 未看到卡牌背景, 1: 看到卡牌背景, 2: 看到卡牌正面
        # WebSocket 服务实例
        self.websocket_server = websocket_server
    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def __del__(self):
        # 关闭套接字
        self.sock.close()

    def send_broadcast_message(self, card1_index, card2_index, port=5005):
        # 发送广播消息
        start = time.time()
        message = f"{card1_index},{card2_index},{start}"
        # if self.websocket_server:
        #     # 使用 call_soon_threadsafe 来安全地调用 asyncio 的协程
        #     self.websocket_server.loop.call_soon_threadsafe(
        #         asyncio.create_task, self.websocket_server.broadcast_message(message)
        #     )
        self.sock.sendto(message.encode(), ('<broadcast>', port))
        self.log(f"广播耗时：{(time.time()-start)*1000:.2f}毫秒 消息: {message}")

    def take_action(self, poker1, poker2):
        self.send_broadcast_message(poker1.classic,poker2.classic)
        start = time.time()
        if poker1.card_num == poker2.card_num:
            key = self.hotkey_he
            name='和'
        elif poker1.card_num > poker2.card_num:
            key = self.hotkey_long
            name = '龙'
        else:
            key = self.hotkey_hu
            name = '虎'
        self.simulate_key_press(key)
        self.log(f"按键耗时:{(time.time()-start)*1000:.2f}毫秒 龙：{poker1.card}，虎：{poker2.card} 下注{name}，按键【{key}】")

    def simulate_key_press(self, key):
        if key:
            pyautogui.press(key)

    def get_white_ratio(self, image, threshold=200):
        """检查图片中是否包含超过指定比例的白色像素"""
        gray = np.mean(image, axis=2)  # 更快地转换为灰度图
        white_pixels = np.count_nonzero(gray > threshold)
        total_pixels = gray.size
        return white_pixels / total_pixels

    def pause(self):
        self.is_paused = True
        self.log("暂停游戏...")

    def resume(self):
        self.is_paused = False
        self.log("继续游戏...")

    def stop(self):
        self.is_running = False
        self.log("停止游戏...")
        # 关闭 ImageProcessor
        self.imageProcessor.stop()

    def run(self,confidence_threshold=0.99):
        if self.show_hint_callback:
            self.show_hint_callback()
        self.log("开始游戏...")
        status = [0,0]
        last_white_ratio = [0.0, 0.0]
        recongnize_cnt = 0
        while self.is_running:
            if self.is_paused:
                time.sleep(0.05)
                continue

            start_time = time.time()
            # 使用 ImageProcessor 处理截图
            white_ratio1, image1, white_ratio2, image2 = self.imageProcessor.process_images()
            screenshot_time = time.time()

            card_front1,card_front2 = self.check_card_background(status, white_ratio1, white_ratio2, image1, image2)
            if not card_front1 and not card_front2:
                continue
            # 检查是否满足所有条件
            if (self.has_seen_card_back[0] and self.has_seen_card_back[1]
                    and self.first_card_back_time[1] is not None
                    and self.first_card_back_time[0] is not None
            ):
                # 使用 ImageProcessor 处理图像识别
                if recongnize_cnt>10 and abs(last_white_ratio[0]-white_ratio1)<0.01 and abs(last_white_ratio[1]-white_ratio2)<0.01 :
                    continue;
                if abs(last_white_ratio[0] - white_ratio1) >= 0.01 or abs(last_white_ratio[1] - white_ratio2) >= 0.01 :
                    recongnize_cnt =0;
                predicted_class1, confidence1, predicted_class2, confidence2 = self.imageProcessor.detect_images(image1, image2)
                detection_time = time.time()
                poker1 = Poker(predicted_class1)
                poker2 = Poker(predicted_class2)

                if confidence1 >= confidence_threshold and confidence2 >= confidence_threshold:
                    recongnize_cnt = 0
                    last_white_ratio = [white_ratio1, white_ratio2]
                    self.log(f"龙{poker1.card} [{confidence1:.4f}]  - 虎{poker2.card} [{confidence2:.4f}] ")
                    self.log(f"截图耗时: {(screenshot_time - start_time) * 1000:.2f} 毫秒")
                    self.log(f"识别图耗时: {(detection_time - screenshot_time) * 1000:.2f} 毫秒")
                    self.log(f"总处理耗时: {(time.time() - start_time) * 1000:.2f} 毫秒")
                    if time.time()-self.first_card_back_time[0]<15.0 and time.time()-self.first_card_back_time[1]<15:
                        self.take_action(poker1, poker2)
                    else:
                        self.log("时间太长，不进行下注")
                    self.update_image_callback(image1, image2, poker1, poker2)
                    # 重置状态变量
                    self.has_seen_card_back = [False, False]
                    self.first_card_back_time = [None, None]
                else:
                    recongnize_cnt+=1
                    last_white_ratio = [white_ratio1,white_ratio2]
                    self.log(f"识别失败，置信度不够 龙{poker1.card} [{confidence1:.4f}]  - 虎{poker2.card} [{confidence2:.4f}] ")
                    self.log(f"截图耗时: {(screenshot_time - start_time) * 1000:.2f} 毫秒")
                    self.log(f"识别图耗时: {(detection_time - screenshot_time) * 1000:.2f} 毫秒")
                    self.log(f"总处理耗时: {(time.time() - start_time) * 1000:.2f} 毫秒")
                    self.update_image_callback(image1, image2, poker1, poker2)
            else:
                self.first_card_back_time = [None, None]
                self.has_seen_card_back = [False, False]

    def _check_card_background(self, status, white_ratio1, white_ratio2, image1, image2):
        old_status0,old_status1 = status[0],status[1]
        if 0.008<white_ratio1<=0.025 and 0.008<white_ratio2<=0.025:
            if status[0]!=1 and status[1]!=1:
                b1, c1, b2, c2 = self.imageProcessor.detect_images_background(image1, image2)
                if c1>0.95 and b1==1 and c2>0.95 and b2==1:
                    self.first_card_back_time[0] = time.time()
                    self.has_seen_card_back[0] = True
                    self.log(f"虎 卡牌背景 {white_ratio2:.4f}")
                    self.first_card_back_time[1] = time.time()
                    self.has_seen_card_back[1] = True
                    self.log(f"龙 卡牌背景 {white_ratio1:.4f}")
                    status[0]=1
                    status[1]=1
                    self.update_image_callback(image1, image2, None, None)

        else:
            if white_ratio1 <= 0.008:
                if status[0] != 0:
                    status[0] = 0
                    self.log(f"龙 无牌 {white_ratio1:.4f}")
            # elif 0.008 < white_ratio1 <= 0.025:
            #     if status[0] != 1:
            #         status[0] = 1
            elif 0.60 <= white_ratio1:
                if status[0] != 2:
                    status[0] = 2
                    self.log(f"龙 其他背景 {white_ratio2:.4f}")

            if 0.025 < white_ratio1 < 0.60:
                if status[0] != 3:
                    status[0] = 3
                    # 卡牌正面
                    self.log(f"龙 卡牌正面 {white_ratio1:.4f}")

            if white_ratio2 <= 0.008:
                if status[1] != 0:
                    status[1] = 0
                    self.log(f"虎 无牌 {white_ratio2:.4f}")
            # elif 0.008 < white_ratio2 <= 0.025:
            #     if status[1] != 1:
            #         status[1] = 1

            elif 0.60 <= white_ratio2:
                if status[1] != 2:
                    status[1] = 2
                    self.log(f"虎 其他背景 {white_ratio2:.4f}")

            if 0.025 < white_ratio2 < 0.60:
                if status[1] != 3:
                    status[1] = 3
                    # 卡牌正面
                    self.log(f"虎 卡牌正面 {white_ratio2:.4f}")

        # if old_status0!=1 and status[0]==1 and old_status1!=1 and status[1]==1:

        card_front1 = 0.025 <= white_ratio1 < 0.60
        card_front2 = 0.025 <= white_ratio2 < 0.60

        return card_front1,card_front2

    def _get_red_ratio(self, image, lower_red1=(0, 100, 100), upper_red1=(10, 255, 255),
                       lower_red2=(160, 100, 100), upper_red2=(180, 255, 255)):
        """统计红色像素占比（HSV颜色空间）"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 定义红色在HSV中的两个范围（色相环首尾）
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        return np.count_nonzero(red_mask) / red_mask.size
    def check_card_background(self, status, white_ratio1, white_ratio2, image1, image2):
        red_ration1 = self._get_red_ratio(image1)
        red_ration2 = self._get_red_ratio(image2)
        if red_ration1>0.20 and red_ration2>0.20 and white_ratio1<=0.063 and white_ratio2<=0.063:
            if status[0]!=1 and status[1]!=1:
                b1, c1, b2, c2 = self.detect_images_background(image1, image2)
                if c1>0.95 and b1==1 and c2>0.95 and b2==1:
                    self.first_card_back_time[0] = time.time()
                    self.has_seen_card_back[0] = True
                    self.logger.info(f"龙 卡牌背面 {white_ratio1:.4f}  置信度:{c1:.4f}")

                    self.first_card_back_time[1] = time.time()
                    self.has_seen_card_back[1] = True
                    self.logger.info(f"虎 卡牌背面 {white_ratio2:.4f}  置信度:{c2:.4f}")

                    status[0]=1
                    status[1]=1
                    # self.update_image_callback(image1, image2, None, None)
        else:
            if white_ratio1 <= 0.01:
                if status[0] != 0:
                    status[0] = 0
                    self.logger.info(f"龙 无牌 {white_ratio1:.4f}")
            elif 0.60 <= white_ratio1:
                if status[0] != 2:
                    status[0] = 2
                    self.logger.info(f"龙 其他背面 {white_ratio2:.4f}")

            if 0.063 < white_ratio1 < 0.60:
                if status[0] != 3:
                    status[0] = 3
                    # 卡牌正面
                    self.logger.info(f"龙 卡牌正面 {white_ratio1:.4f}")

            if white_ratio2 <= 0.01:
                if status[1] != 0:
                    status[1] = 0
                    self.logger.info(f"虎 无牌 {white_ratio2:.4f}")
            # elif 0.008 < white_ratio2 <= 0.025:
            #     if status[1] != 1:
            #         status[1] = 1

            elif 0.60 <= white_ratio2:
                if status[1] != 2:
                    status[1] = 2
                    self.logger.info(f"虎 其他背面 {white_ratio2:.4f}")

            if 0.063 < white_ratio2 < 0.60:
                if status[1] != 3:
                    status[1] = 3
                    # 卡牌正面
                    self.logger.info(f"虎 卡牌正面 {white_ratio2:.4f}")

        # if old_status0!=1 and status[0]==1 and old_status1!=1 and status[1]==1:

        card_front1 = 0.063 <= white_ratio1 < 0.60 and white_ratio1>red_ration1
        card_front2 = 0.063 <= white_ratio2 < 0.60 and white_ratio2>red_ration2

        return card_front1,card_front2