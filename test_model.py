from ultralytics import YOLO
import cv2
print("Program started")
# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")
print("Model loaded")
print("Classes:", model.names)
# Run detection
results = model.predict(
    source="img23.jpeg",
    conf=0.017,
    save=True
)
print("Prediction finished")
for r in results:
    print("Detections:", len(r.boxes))
    im = r.plot(labels=False, conf=False)
    cv2.imshow("Detection", im)
    cv2.waitKey(0)