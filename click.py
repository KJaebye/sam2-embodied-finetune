import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # Turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

class InteractiveSegmentation:
    def __init__(self, predictor, image):
        self.predictor = predictor
        self.image = image
        self.point_coords = []
        self.input_labels = []
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(self.image)
        # Connect events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.axis('off')
        plt.show()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        if event.key == 'control':  # Ctrl + 鼠标点击选择背景点
            self.point_coords.append([x, y])
            self.input_labels.append(0)  # 背景点
        else:  # 普通点击选择前景点
            self.point_coords.append([x, y])
            self.input_labels.append(1)  # 前景点
        self.update_display()

    def on_key(self, event):
        if event.key == 'escape':  # 按 Esc 键取消上一次点击的点
            if self.point_coords:
                self.point_coords.pop()
                self.input_labels.pop()
                self.update_display()

    def update_display(self):
        self.ax.clear()
        self.ax.imshow(self.image)
        if self.point_coords:  # 如果有输入点，才进行预测
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.predictor.set_image(self.image)
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(self.point_coords),
                    point_labels=np.array(self.input_labels),
                    multimask_output=True,
                )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            if len(masks) > 0:
                show_mask(masks[0], self.ax)
            show_points(np.array(self.point_coords), np.array(self.input_labels), self.ax)
        self.fig.canvas.draw()

if __name__ == "__main__":
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # 使用 OpenCV 加载图像
    image = cv2.imread("assets/7.png")  # 支持 JPG、PNG、BMP 等格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # 启动交互式分割
    InteractiveSegmentation(predictor, image)