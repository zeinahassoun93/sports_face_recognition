from ultralytics import YOLO
import cv2
import os

if __name__ == "__main__":

    model = YOLO("runs/detect/train6/weights/best.pt")  # load a custom trained model

    test_images_dir = "dataset/Ronaldo-x-Messi-Detection-1/test/images"

    output_dir = "runs/detect/test_results"

    results = model.predict(
        source = test_images_dir,
        conf = 0.5,
        save = True,
        imgsz = 640,
        device= 0
    )  

print("Inference completed. Results saved to:", output_dir)
print("Results saved", results[0].save_dir)

