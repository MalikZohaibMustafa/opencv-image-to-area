import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def process_image(image, width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None
    results = []

    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue

        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if pixelsPerMetric is None:
            pixelsPerMetric = dB / width

        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric
        area = cv2.contourArea(c) / (pixelsPerMetric ** 2)

        cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(orig, "Area: {:.2f}".format(area), (int(tltrX - 15), int(tltrY + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.png', orig)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        results.append({
            "contour_number": len(results) + 1,
            "width": dimB,
            "height": dimA,
            "area": area,
            "image": image_base64
        })

    return results

@app.route('/process_image', methods=['POST'])
def process_image_api():
    if 'image' not in request.files or 'width' not in request.form:
        return jsonify({"error": "Invalid input"}), 400

    image_file = request.files['image']
    width = float(request.form['width'])

    image = Image.open(image_file)
    image = np.array(image)
    print("image recieved", width)

    results = process_image(image, width)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
