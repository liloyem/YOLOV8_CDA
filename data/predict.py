from ultralytics import YOLO
import cv2
import torch
from ultralytics.nn.tasks import DetectionModel


torch.serialization.add_safe_globals([DetectionModel])


model = YOLO('D:/AI/yolov8-main/Yolov8n_CDA.pt')

#model = YOLO('D:/AI/yolov8-main/Yolov8n_best.pt')


results = model.predict(
    source='D:/AI/yolov8-main/2.jpg',
    conf=0.5,
    save=False,
    save_txt=False,
    show=False,
    project='runs/detect',
    name='predict'
)


result = results[0]
print(f"检测到 {len(result.boxes)} 个对象：")
for i, box in enumerate(result.boxes):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    confidence = box.conf[0].item()
    class_id = int(box.cls[0].item())
    label = model.names[class_id]
    print(f"  对象 {i+1}: {label}, 置信度: {confidence:.2f}, 坐标: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")


annotated_image = result.plot()

annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

cv2.imshow('YOLOv8 Detection', annotated_image)


cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv8 Detection', 640, 640)

print("\n按任意键关闭图像窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()


output_path = 'D:/AI/yolov8-main/detection_result31.jpg'
cv2.imwrite(output_path, annotated_image)
print(f"检测结果已保存至: {output_path}")