from ultralytics import YOLO

if __name__ == '__main__':
# load the YOLOv8 model
    model = YOLO("yolov8s.pt")


    # train my model

    model.train(data="dataset/Ronaldo-x-Messi-Detection-1/data.yaml",
                epochs=100,
                imgsz=640,
                batch=16)
