from ultralytics import YOLO
import time

if __name__ == '__main__':

    model = YOLO('runs/detect/train42/weights/best.pt')

    #results = model.predict(source = 'data/images/train', conf = 0.5, save=True)
    results = model.predict(source = 'download.jpg', conf = 0.5, save=True, show = True, line_width = 1)
    #results = model.predict(source = 'production_id_4068820.mp4', conf = 0.5, save=True, show = True, line_width = 2)
    time.sleep(1)
    #results = model.val(source = 'data')