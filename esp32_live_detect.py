import cv2
import time
from ultralytics import YOLO

model = YOLO("best.pt")

url = "http://10.217.121.80:81/stream"
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_detection_time = 0
interval = 3

while True:
    ret, frame = cap.read()

    if not ret:
        print("Retrying...")
        continue

    current_time = time.time()

    if current_time - last_detection_time >= interval:
        results = model.predict(frame, conf=0.25)

        count = 0
        for r in results:
            count = len(r.boxes)
            frame = r.plot(labels=False, conf=False)

        last_detection_time = current_time

    cv2.putText(frame, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("ESP32 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()