from scipy.spatial import distance as dist
import argparse
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def show_image(title, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def detect_points(image, lower_color_bound, upper_color_bound):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color_bound, upper_color_bound)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    points = []
    for c in cnts:
        if cv2.contourArea(c) > 5:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                points.append((cX, cY))
    return points

def sort_points(points):
    points = sorted(points, key=lambda x: x[1])  # Sort by y-coordinate
    top_points = sorted(points[:2], key=lambda x: x[0])  # Sort top points by x-coordinate
    bottom_points = sorted(points[2:], key=lambda x: x[0])  # Sort bottom points by x-coordinate
    return top_points + bottom_points

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# Define the color range for detecting points (example for red color)
lower_color_bound = (0, 100, 100)
upper_color_bound = (10, 255, 255)

points = detect_points(image, lower_color_bound, upper_color_bound)
print(f"Detected points: {points}")

if len(points) != 4:
    print("Error: Exactly four points are required to form a rectangle.")
    exit(1)

sorted_points = sort_points(points)
print(f"Sorted points: {sorted_points}")

(tl, tr, br, bl) = sorted_points

# Draw the points on the image
for (x, y) in [tl, tr, br, bl]:
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# Calculate the midpoints and distances
(tltrX, tltrY) = midpoint(tl, tr)
(blbrX, blbrY) = midpoint(bl, br)
(tlblX, tlblY) = midpoint(tl, bl)
(trbrX, trbrY) = midpoint(tr, br)

cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

pixelPerMetric = dB / args["width"] if dB != 0 else None

if pixelPerMetric is None or pixelPerMetric == 0:
    print("Error in calculating pixelPerMetric. Exiting.")
    exit(1)

dimA = dA / pixelPerMetric
dimB = dB / pixelPerMetric

print(f"Width = {dimB:.1f} in, Height = {dimA:.1f} in")

cv2.putText(image, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
cv2.putText(image, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

show_image("Image with measurements", image)
