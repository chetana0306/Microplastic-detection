import cv2
import time
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
url = "http://10.217.121.80:81/stream"
cap = cv2.VideoCapture(url)
last_detection_time = 0
interval = 3   # seconds
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    current_time = time.time()
    # run detection every 3 seconds
    if current_time - last_detection_time >= interval:
        results = model.predict(frame, conf=0.25)
        for r in results:
            frame = r.plot(labels=False, conf=False)
        last_detection_time = current_time
    cv2.imshow("ESP32 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()