import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/ADAM=V8+URPC/weights/best.pt')
    model.val(data='dataset/URPC.yaml',
                split='test',
                save_json=True, # if you need to cal coco metrice
                project='out',
                name='URPC-V8-ADAMA',
                )