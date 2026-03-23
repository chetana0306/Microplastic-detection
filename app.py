from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load trained model
model = YOLO("best.pt")

# ESP32 stream URL
ESP32_URL = "http://10.217.121.80:81/stream"

# Open stream
cap = cv2.VideoCapture(ESP32_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

last_detection_time = 0
interval = 2  # seconds
count = 0

def generate_frames():
    global last_detection_time, count

    while True:
        success, frame = cap.read()

        if not success:
            continue

        current_time = time.time()

        # Run YOLO every 2 sec
        if current_time - last_detection_time >= interval:
            results = model.predict(frame, conf=0.25)

            for r in results:
                count = len(r.boxes)
                frame = r.plot(labels=False, conf=False)

            last_detection_time = current_time

        # Show count on frame
        cv2.putText(frame, f"Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Microplastic Live Detection</title>
    </head>
    <body style="background:#0f2027; color:white; text-align:center;">
        <h1>🔬 Live Microplastic Detection</h1>
        <img src="/video" width="700">
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)