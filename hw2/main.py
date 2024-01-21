import cv2
import numpy as np

img = cv2.imread('./test1.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

edges = cv2.Canny(image=img_blur, threshold1=60, threshold2=120)
kernel = np.ones((1, 1), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=1)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if len(contour) < 10:
        continue
    points = contour.reshape(-1, 2)
    ellipse = cv2.fitEllipse(points)
    _, axes, _ = ellipse
    major_axis, minor_axis = axes
    if abs(major_axis) > abs(minor_axis):
        e = abs(major_axis) / abs(minor_axis)
    else:
        e = abs(minor_axis) / abs(major_axis)
    if e < 1.05:
        continue
    cv2.ellipse(img, ellipse, (0, 0, 255), 2)

cv2.imshow('Ellipse', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
