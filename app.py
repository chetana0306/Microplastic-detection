from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

if not os.path.exists("static"):
    os.makedirs("static")

global_data = {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["image"]

    image = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _, thresh = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    particle_count = 0
    particle_data = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if 400 < area < 5000:

            particle_count += 1

            x,y,w,h = cv2.boundingRect(contour)

            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

            cv2.putText(
                image,
                f"P{particle_count}",
                (x,y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                1
            )

            particle_data.append({
                "Particle_ID":particle_count,
                "Area":area
            })

    df = pd.DataFrame(particle_data)

    if len(df)>0:
        avg_area = round(df["Area"].mean(),2)
        max_area = round(df["Area"].max(),2)
        min_area = round(df["Area"].min(),2)
    else:
        avg_area = max_area = min_area = 0

    if particle_count < 5:
        contamination = "LOW"
        color = "green"
    elif particle_count < 15:
        contamination = "MEDIUM"
        color = "orange"
    else:
        contamination = "HIGH"
        color = "red"

    output_path = "static/result.jpg"
    cv2.imwrite(output_path,image)

    graph_path = "static/graph.png"

    if len(df)>0:
        plt.figure(figsize=(6,4))
        plt.hist(df["Area"],bins=10,color="skyblue",edgecolor="black")
        plt.title("Particle Size Distribution")
        plt.xlabel("Particle Area")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(graph_path)
        plt.close()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    global global_data
    global_data = {
        "count":particle_count,
        "avg":avg_area,
        "max":max_area,
        "min":min_area,
        "level":contamination,
        "time":now,
        "image":output_path,
        "graph":graph_path
    }

    table = df.to_dict(orient="records")

    return render_template(
        "result.html",
        count=particle_count,
        contamination=contamination,
        color=color,
        avg_area=avg_area,
        max_area=max_area,
        min_area=min_area,
        table=table,
        image_path=output_path,
        graph_path=graph_path
    )


@app.route("/report")
def report():

    styles = getSampleStyleSheet()

    pdf_path = "static/microplastic_report.pdf"

    story = []

    story.append(Paragraph("Microplastic Detection Report", styles['Title']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"Date & Time: {global_data['time']}", styles['Normal']))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"Particles Detected: {global_data['count']}", styles['Normal']))
    story.append(Paragraph(f"Contamination Level: {global_data['level']}", styles['Normal']))
    story.append(Paragraph(f"Average Particle Size: {global_data['avg']}", styles['Normal']))
    story.append(Paragraph(f"Largest Particle: {global_data['max']}", styles['Normal']))
    story.append(Paragraph(f"Smallest Particle: {global_data['min']}", styles['Normal']))

    story.append(Spacer(1,30))

    story.append(Paragraph("Detected Image", styles['Heading2']))
    story.append(Image(global_data["image"], width=400, height=250))

    story.append(Spacer(1,30))

    story.append(Paragraph("Particle Size Graph", styles['Heading2']))
    story.append(Image(global_data["graph"], width=400, height=250))

    doc = SimpleDocTemplate(pdf_path)
    doc.build(story)

    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)