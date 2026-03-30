import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\bishe\ultralytics-main\runs\train\exp18\weights\last.pt')
    model.val(data=r'E:/bishe/ultralytics-main/data444.yaml',
              split='val',
              imgsz=640,
              batch=16,
              iou=0.6,
              rect=False,
              save_json=False,
              project='runs/val',
              name='exp',
              )