import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/yolov5-C3-Faster-LSKASPPF.yaml/weights/best.pt')
    model.track(source='test2.mp4',
                project='runs/track',
                name='yolov5-C3-Faster-LSKASPPF.yaml',
                save=True
                )