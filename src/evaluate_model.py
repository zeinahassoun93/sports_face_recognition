from ultralytics import YOLO
import numpy as np


if __name__ == '__main__':
    model = YOLO("runs/detect/train6/weights/best.pt")  # load a custom trained model

    metrics = model.val()

    print("Evaluation Metrics:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {np.mean(metrics.box.p):.4f}")
    print(f"Recall:    {np.mean(metrics.box.r):.4f}")

    # Per CLass results
    print("\nPer Class Results:")
    for cls, value in metrics.results_dict.items():
        print(f"{cls}:{value}")