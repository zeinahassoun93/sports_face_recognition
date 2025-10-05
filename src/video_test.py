from ultralytics import YOLO
import cv2
import os

if __name__ == "__main__":

    model = YOLO("runs/detect/train6/weights/best.pt") 

    video_path = "dataset/Ronaldo-x-Messi-Detection-1/videoplayback.mp4"
    output_path = "runs/detect/predict/videos/output.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Error: Could not open video.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("Processing video...Press 'q' to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.5,imgsz=640, device = 0, verbose=False)
        result_frame = results[0].plot()
        cv2.imshow("YOLOv8 Detection", result_frame)
        out.write(result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to {output_path}")