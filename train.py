import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')
    #model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='dataset/URPC.yaml',
                cache=False,
                project='runs/train',
                name='ADAM=V8+URPC',
                epochs=500,
                workers=16,
                batch=32,
                close_mosaic=10,
                #hsv_h=0.015,
                #hsv_s=0.7,
                #copy_paste=0.5,
                optimizer='Adam', # using SGD
                #resume='runs/train/SLM-AFPN-C2f_SCConv-PASCAL2/weights/last.pt', # last.pt path
                # amp=False # close amp
                # fraction=0.2
                patience=50
                )