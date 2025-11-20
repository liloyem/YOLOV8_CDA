import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential

# 添加所有必要的安全全局变量
torch.serialization.add_safe_globals([
    DetectionModel,
    Sequential
])

from ultralytics import YOLO

# 加载模型
model = YOLO('D:/AI/yolov8-main/CDA_best.pt')
model.eval()  # 设置为评估模式

# 2. 定义TargetLayer（目标层）
# YOLOv8n的最后一个卷积层是 `model.model.model[-2]`
# 但结构可能因版本和自定义而异，建议打印模型结构确认
target_layers = [model.model.model[-2]]

# 3. 初始化GradCAM
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True) # 如果使用GPU，改为True

# 4. 准备输入图像
image_path = "D:/AI/yolov8-main/test_image.jpg"
rgb_img = cv2.imread(image_path, 1)
rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
img_tensor = cv2.resize(rgb_img, (640, 640))  # 调整为你模型的输入尺寸
img_tensor = img_tensor / 255.0  # 归一化
img_tensor = torch.from_numpy(img_tensor).float().permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]

# 5. 指定目标类别（可选）
# 如果你的图中有多个类别，可以指定你要生成热力图的类别索引
# 例如，如果你的'crack'类在数据集中的索引是0，则指定 targets = [0]
targets = None  # None则会选择模型预测得分最高的那个类别

# 6. 生成热力图
grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]  # 取出第一个（也是唯一一个）batch的热力图

# 7. 将热力图叠加到原图上并可视化
visualization = show_cam_on_image(rgb_img.astype(np.float32) / 255.0,
                                  grayscale_cam,
                                  use_rgb=True)

# 8. 保存或显示结果
cv2.imwrite('heatmap_output.jpg', visualization)