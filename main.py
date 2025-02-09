import asyncio
import threading

from PIL import ImageTk
import os
import keyboard
from tkinter import messagebox
import configparser
from game_controller import GameController
import logging
import tkinter as tk
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from websocket_server import WebSocketServer

os.environ['PYTHONIOENCODING'] = 'utf-8'

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),  # 指定文件编码为 UTF-8
        logging.StreamHandler()         # 输出到控制台
    ]
)

class GUI:
    def __init__(self, root, loop):
        self.root = root
        self.root.title("Poker Game Controller")
        self.loop = loop

        # 读取配置文件
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        if not self.config.has_section('Settings'):
            self.config.add_section('Settings')
            self.config.set('Settings', 'long_x', '1437')
            self.config.set('Settings', 'long_y', '883')
            self.config.set('Settings', 'width', '54')
            self.config.set('Settings', 'distance', '146')
            self.config.set('Settings', 'hotkey_long', 'num1')
            self.config.set('Settings', 'hotkey_hu', 'num2')
            self.config.set('Settings', 'hotkey_he', 'num3')
            self.config.set('Settings', 'images_path', 'images')
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)

        image_folder = self.config.get('Settings', 'images_path')
        os.makedirs(image_folder, exist_ok=True)

        # 创建和布局控件
        self.create_widgets()

        # 初始化日志文本框
        self.log_text = tk.Text(self.root, height=10, width=60)
        self.log_text.grid(row=6, column=0, columnspan=4, padx=10, pady=10)

        # 初始化图片标签
        self.image_label1 = tk.Label(self.root)
        self.image_label1.grid(row=7, column=0, padx=10, pady=10)
        self.image_label2 = tk.Label(self.root)
        self.image_label2.grid(row=7, column=1, padx=10, pady=10)

        self.result_label1 = tk.Label(self.root)
        self.result_label1.grid(row=8, column=0, padx=10, pady=5)
        self.result_label2 = tk.Label(self.root)
        self.result_label2.grid(row=8, column=1, padx=10, pady=5)

        # 初始化提示信息标签
        self.hint_label = tk.Label(self.root, text="按'F2'暂停 | 'F3'继续 | 'Esc'停止", justify=tk.LEFT)
        self.hint_label.grid(row=9, column=0, columnspan=4, padx=10, pady=10)

        # 初始化游戏控制器实例
        self.game = None

        keyboard.add_hotkey('esc', self.on_esc)
        keyboard.add_hotkey('f2', self.on_f2)
        keyboard.add_hotkey('f3', self.on_f3)
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.websocket_server = WebSocketServer(loop=self.loop)
        # 启动 WebSocket 服务器
        self.start_websocket_server()

    def start_websocket_server(self):
        self.log("WebSocket 没有启动服务")
        #不要开启
        #self.websocket_task = self.loop.create_task(self.websocket_server.start())

    def on_esc(self):
        if self.game is not None:
            self.game.stop()
        # 禁用启动按钮
        # 游戏结束后启用启动按钮
        self._enable_start_button()
        self.game=None

    def on_f2(self):
        if self.game is not None:
            self.game.pause()

    def on_f3(self):
        if self.game is not None:
            self.game.resume()

    def create_widgets(self):
        # 选择截图区域按钮
        self.select_region_button = tk.Button(self.root, text="选择截图区域", command=self.select_screenshot_region)
        self.select_region_button.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

        # X
        self.long_x_label = tk.Label(self.root, text="龙扑克左上角坐标X:")
        self.long_x_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
        self.long_x_entry = tk.Entry(self.root, width=10)
        self.long_x_entry.grid(row=1, column=1, padx=10, pady=5)
        self.long_x_entry.insert(0, self.config.get('Settings', 'long_x'))

        # Y
        self.long_y_label = tk.Label(self.root, text="龙扑克左上角坐标Y:")
        self.long_y_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
        self.long_y_entry = tk.Entry(self.root, width=10)
        self.long_y_entry.grid(row=2, column=1, padx=10, pady=5)
        self.long_y_entry.insert(0, self.config.get('Settings', 'long_y'))

        # 宽度
        self.width_label = tk.Label(self.root, text="截取图像宽度:")
        self.width_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
        self.width_entry = tk.Entry(self.root, width=10)
        self.width_entry.grid(row=3, column=1, padx=10, pady=5)
        self.width_entry.insert(0, self.config.get('Settings', 'width'))

        # 距离
        self.distance_label = tk.Label(self.root, text="虎扑克距离龙左上角距离:")
        self.distance_label.grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
        self.distance_entry = tk.Entry(self.root, width=10)
        self.distance_entry.grid(row=4, column=1, padx=10, pady=5)
        self.distance_entry.insert(0, self.config.get('Settings', 'distance'))

        # 龙下注热键
        self.hotkey_long_label = tk.Label(self.root, text="龙下注热键:")
        self.hotkey_long_label.grid(row=1, column=2, padx=10, pady=5, sticky=tk.W)
        self.hotkey_long_entry = tk.Entry(self.root, width=10)
        self.hotkey_long_entry.grid(row=1, column=3, padx=10, pady=5)
        self.hotkey_long_entry.insert(0, self.config.get('Settings', 'hotkey_long'))

        # 虎下注热键
        self.hotkey_hu_label = tk.Label(self.root, text="虎下注热键:")
        self.hotkey_hu_label.grid(row=2, column=2, padx=10, pady=5, sticky=tk.W)
        self.hotkey_hu_entry = tk.Entry(self.root, width=10)
        self.hotkey_hu_entry.grid(row=2, column=3, padx=10, pady=5)
        self.hotkey_hu_entry.insert(0, self.config.get('Settings', 'hotkey_hu'))

        # 和下注热键
        self.hotkey_he_label = tk.Label(self.root, text="和下注热键:")
        self.hotkey_he_label.grid(row=3, column=2, padx=10, pady=5, sticky=tk.W)
        self.hotkey_he_entry = tk.Entry(self.root, width=10)
        self.hotkey_he_entry.grid(row=3, column=3, padx=10, pady=5)
        self.hotkey_he_entry.insert(0, self.config.get('Settings', 'hotkey_he'))

        # 启动按钮
        self.start_button = tk.Button(self.root, text="启动", command=self.start_game)
        self.start_button.grid(row=5, column=0, columnspan=4, pady=10)

    def log(self, message):
        # 将日志消息发送到日志记录器
        self.loop.call_soon_threadsafe(self._log, message)

    def _log(self, message):
        # 将日志消息发送到日志记录器
        logging.info(message)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

        # 限制日志条数，例如保留最近的500条
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete(1.0, 101.0)  # 删除最早的一行


    def update_image(self, image1, image2, poker1, poker2):
        self.loop.call_soon_threadsafe(self._update_image, image1, image2, poker1, poker2)

    def _update_image(self, image1, image2, poker1, poker2):
        photo1 = ImageTk.PhotoImage(image1)
        photo2 = ImageTk.PhotoImage(image2)

        self.image_label1.config(image=photo1)
        self.image_label1.image = photo1

        self.image_label2.config(image=photo2)
        self.image_label2.image = photo2

        self.result_label1.config(text=f"龙: {poker1.card if poker1 else '?'}")
        self.result_label2.config(text=f"虎: {poker2.card if poker2 else '?'}")
        if poker1 and poker2:
            now = datetime.now()
            image_folder = self.config.get('Settings', 'images_path')
            # 格式化日期和时间
            date_str = now.strftime("%Y%m%d")
            hour_str = now.strftime("%H")
            formatted_time = now.strftime("%Y%m%d%H%M%S.%f")[:-3]
            # 创建子文件夹路径
            subfolder_path = os.path.join(image_folder, date_str, hour_str)
            # 检查并创建子文件夹
            os.makedirs(subfolder_path, exist_ok=True)
            # 保存图片到子文件夹
            image_path1 = os.path.join(subfolder_path, f"{formatted_time}_{poker1.num}_龙.png")
            image1.save(image_path1)
            image_path2 = os.path.join(subfolder_path, f"{formatted_time}_{poker2.num}_虎.png")
            image2.save(image_path2)
        else:
            now = datetime.now()
            image_folder = self.config.get('Settings', 'images_path')
            # 格式化日期和时间
            date_str = now.strftime("%Y%m%d")
            hour_str = now.strftime("%H")
            formatted_time = now.strftime("%Y%m%d%H%M%S.%f")[:-3]
            # 创建子文件夹路径
            subfolder_path = os.path.join(image_folder, date_str, hour_str)
            # 检查并创建子文件夹
            os.makedirs(subfolder_path, exist_ok=True)
            # 保存图片到子文件夹
            image_path1 = os.path.join(subfolder_path, f"{formatted_time}_龙.png")
            image1.save(image_path1)
            image_path2 = os.path.join(subfolder_path, f"{formatted_time}_虎.png")
            image2.save(image_path2)

    def start_game(self):
        if self.game is not None:
            messagebox.showwarning("警告", "游戏已经在运行中")
            return

        try:
            x = int(self.long_x_entry.get())
            y = int(self.long_y_entry.get())
            width = int(self.width_entry.get())
            distance = int(self.distance_entry.get())
            hotkey_long = self.hotkey_long_entry.get()
            hotkey_hu = self.hotkey_hu_entry.get()
            hotkey_he = self.hotkey_he_entry.get()

            self.log(f"读取到的配置: X={x}, Y={y}, 宽度={width}, 距离={distance}, 龙下注热键={hotkey_long}, 虎下注热键={hotkey_hu}, 和下注热键={hotkey_he}")

            # 保存配置到 config.ini
            self.config.set('Settings', 'long_x', str(x))
            self.config.set('Settings', 'long_y', str(y))
            self.config.set('Settings', 'width', str(width))
            self.config.set('Settings', 'distance', str(distance))
            self.config.set('Settings', 'hotkey_long', hotkey_long)
            self.config.set('Settings', 'hotkey_hu', hotkey_hu)
            self.config.set('Settings', 'hotkey_he', hotkey_he)
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)

            # 创建游戏控制器实例
            self.game = GameController(x=x, y=y, width=width, distance=distance, hotkey_long=hotkey_long, hotkey_hu=hotkey_hu, hotkey_he=hotkey_he, log_callback=self.log, update_image_callback=self.update_image, websocket_server=self.websocket_server)

            # 禁用启动按钮
            self.start_button.config(state=tk.DISABLED)

            # 使用线程运行游戏控制器
            game_thread = threading.Thread(target=self.run_game)
            game_thread.daemon = True
            game_thread.start()
        except ValueError:
            messagebox.showerror("输入错误", "请确保 X、Y、宽度和距离是整数")

    def run_game(self):
        # 使用 fallback 参数设置默认值
        confidence_threshold = self.config.getfloat('Settings', 'confidence_threshold', fallback=0.99)
        self.game.run(confidence_threshold)
        # 游戏结束后启用启动按钮
        self.loop.call_soon_threadsafe(self._enable_start_button)

    def _enable_start_button(self):
        self.start_button.config(state=tk.NORMAL)

    def select_screenshot_region(self):
        # 创建一个新的窗口来选择区域
        select_window = tk.Toplevel(self.root)
        select_window.title("选择截图区域")
        select_window.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        select_window.attributes('-fullscreen', True)
        select_window.attributes('-alpha', 0.5)  # 设置窗口透明度

        def on_mouse_down(event):
            self.selection_start = (event.x, event.y)

        def on_mouse_up(event):
            self.selection_end = (event.x, event.y)
            select_window.destroy()
            self.process_selection()

        def on_mouse_move(event):
            if self.selection_start:
                canvas.delete("selection")
                canvas.create_rectangle(self.selection_start[0], self.selection_start[1], event.x, event.y,
                                        outline="red", tag="selection")

        canvas = tk.Canvas(select_window, bg="white")
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<Button-1>", on_mouse_down)
        canvas.bind("<ButtonRelease-1>", on_mouse_up)
        canvas.bind("<B1-Motion>", on_mouse_move)

    def process_selection(self):
        if self.selection_start and self.selection_end:
            left = min(self.selection_start[0], self.selection_end[0])
            top = min(self.selection_start[1], self.selection_end[1])
            width = abs(self.selection_end[0] - self.selection_start[0])
            height = abs(self.selection_end[1] - self.selection_start[1])

            # 显示选择的区域
            selected_region = (left, top, width, height)
            confirm = messagebox.askyesno("选择区域", f"选择的区域: {selected_region},确定要应用这个配置吗？")

            if confirm:
                # 更新输入框中的值
                self.long_x_entry.delete(0, tk.END)
                self.long_x_entry.insert(0, str(left))
                self.long_y_entry.delete(0, tk.END)
                self.long_y_entry.insert(0, str(top))
                self.width_entry.delete(0, tk.END)
                self.width_entry.insert(0, str(width if width >= height else height))
                # self.distance_entry.delete(0, tk.END)
                # self.distance_entry.insert(0, str(height))

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    root = tk.Tk()
    app = GUI(root, loop)

    def run_asyncio():
        loop.run_forever()

    # 创建一个线程运行 asyncio
    asyncio_thread = threading.Thread(target=run_asyncio)
    asyncio_thread.daemon = True
    asyncio_thread.start()

    # 运行 tkinter 的 mainloop
    root.mainloop()

if __name__ == "__main__":
    main()
