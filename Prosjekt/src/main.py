import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

filein = "./Filtered_sudoku_Images/Filtered_1_sudoku.jpg"

# white color mask
img = cv2.imread(filein)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 360  # angular resolution in radians of the Hough grid (180)
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 150  # minimum number of pixels making up a line (50)
max_line_gap = 15  # maximum gap in pixels between connectable line segments (20)
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

min_length = 200
filtered_lines = []

for line in lines:
    x1,y1,x2,y2 = line[0]
    line_length = np.linalg.norm([x2-x1, y2-y1])
    if line_length < min_length:
        filtered_lines.append(line)

line_filtered_image = np.zeros_like(img) # helt hvit bilde

for line in filtered_lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(line_filtered_image, (x1,y1), (x2,y2), (0, 255, 0), 2)

lines_edges_filtered = cv2.addWeighted(img, 0.8, line_filtered_image, 1, 0)

lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

plt.imshow(lines_edges_filtered)
plt.title("Filtered lines")
plt.show()

plt.imshow(lines_edges)
plt.title("Lines sudoku")
plt.show()