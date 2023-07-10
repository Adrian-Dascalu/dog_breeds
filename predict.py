from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('runs/detect/train32/weights/best.pt')

    results = model.predict(source = 'data/images/train', conf = 0.5, save=True)
    #results = model.predict(source = 'pexels-shvets-production-7546822(1080p).mp4', conf = 0.5, save=True, show = True)
    #results = model.val(source = 'data')