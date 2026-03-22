# Microplastic-detection
 Microplastic Detection System
📖 Overview
This project presents a real-time microplastic detection system using ESP32-CAM, YOLO (You Only Look Once) deep learning, and a Flask web application. The system captures water sample images, processes them using a trained model, and displays detection results through a browser interface.

🎯 Objectives
Capture real-time images using ESP32-CAM

Detect microplastics using YOLO model

Display results on a web interface

Analyze contamination levels

🏗️ System Architecture
ESP32-CAM → Image Stream → YOLO Detection → Flask Server → Web Interface

⚙️ Technologies Used
Python, OpenCV

YOLO (Ultralytics)

Flask

NumPy, Pandas, Matplotlib

HTML/CSS

🔧 Hardware Components
ESP32-CAM

Water Pump

MOSFET (IRF/IRL)

Voltage Regulator (7805/7809)

LEDs, Breadboard, Power Supply

💻 Working Principle
The ESP32-CAM streams live images over WiFi. These frames are processed using a YOLO model that identifies microplastic particles. The processed frames are then sent to a Flask server, which streams the output to a web interface. The system also calculates particle count and determines contamination level (Low, Medium, High) based on detected particles.

🚀 Features
Real-time detection

Live web streaming

Particle analysis

Contamination level classification

Graph and report generation

▶️ Usage
Run Flask server: python app.py

Run YOLO detection script

Open browser at http://127.0.0.1:5000

📊 Output
Detected image/video

Particle count

Contamination level

Graph visualization

PDF report

🔮 Future Scope
Cloud deployment

Mobile app integration

Improved detection accuracy

Automated filtration system

👩‍💻 Team
Snehitha – Detection
Chetana – Web
Bhavani – Hardware

📌 Conclusion
This project integrates IoT, AI, and web technologies to provide an efficient and low-cost solution for monitoring microplastic contamination in water.
