import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/AI/yolov8-main/ultralytics/cfg/models/v8/Yolov8_CDA.yaml', task='detect')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=r'D:/AI/yolov8-main/ultralytics/cfg/datasets/fall.yaml',
                cache=False,  # 禁用缓存以避免内存问题
                imgsz=640,
                epochs=200,
                single_cls=False,
                batch=16,  # 4090可以处理更大的batch size
                workers=8,  # 增加workers以提高数据加载速度
                device='0',  # 使用第一个GPU (4090)
                optimizer='SGD',
                # resume = True,
                exist_ok=True,
                project='runs/train_CDA',
                name='exp'
                )
    #
    #     epochs=300,
    #     batch=16,
    #     workers=8,
    #     device='0',
    #     optimizer='AdamW',
    #     lr0=0.001,
    #     lrf=0.01,
    #     momentum=0.9,
    #     weight_decay=0.05,
    #     warmup_epochs=5.0,
    #     warmup_momentum=0.8,
    #     warmup_bias_lr=0.1,
    #     box=7.5,
    #     cls=0.5,
    #     dfl=1.5,
    #     fl_gamma=1.5,  # 焦点损失
    #     label_smoothing=0.1,  # 标签平滑
    #     hsv_h=0.015,
    #     hsv_s=0.7,
    #     hsv_v=0.4,
    #     degrees=10.0,
    #     translate=0.1,
    #     scale=0.5,
    #     shear=2.0,
    #     perspective=0.001,
    #     flipud=0.0,
    #     fliplr=0.5,
    #     mosaic=1.0,
    #     mixup=0.1,
    #     copy_paste=0.1,
    #     close_mosaic=15,
    #     cos_lr=True,  # 余弦学习率
    #     val=True,
    #     save=True,
    #     save_period=10,
    #     plots=True,
    #     save_json=True,
    #     cache=False,
    #     single_cls=False,
    #     verbose=True,
    #     exist_ok=True,
    #     project='runs/cda',
    #     name='exp1'
    # )