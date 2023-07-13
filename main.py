#if __name__ == '__main__':
#def main():

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("runs/detect/train38/weights/best.pt") # 

    result = model.train(data = "config.yaml", epochs = 200, device = 0, amp = False, workers = 2)#amp = False, batch = 4, workers = 3

#if __name__ == '__main__':
 #   result = main()