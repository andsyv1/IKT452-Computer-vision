import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from CNN_filter_class import CNN_filter
from Neural import CNN
import matplotlib.pyplot as plt

class SudokuVision():

    def __init__(self, image, cnn_model):
        self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        self.cnn_model = cnn_model
        self.corners = self.shi_thomas()
        self.filtered_corners = self.filter_lines()
        self.edges = self.edgesmap()
        self.lines = self.hough()
        self.cells = self.extract_cells()
        

    def shi_thomas(self): # Shi Thomas algorithm for finding corners
        corners = cv2.goodFeaturesToTrack(self.image, 800, 0.0002, 0.001)
        corners = np.intp(corners)
        return corners
    

    def edgesmap(self): # Making an edge map so it is sutibel for the Hogh transform and making corners appere whit red circels
        edge_map = cv2.Canny(self.image, 50, 150) # PRØVE Å ENDRE PÅ SIGMA...
        egde_map_black = edge_map.copy()
        for corner in self.corners:
            x, y = corner.ravel()
            cv2.circle(egde_map_black, (x, y), 1, (0, 0, 255), -1)

        return egde_map_black

    
    def filter_lines(self):
        grid = np.array([c.ravel() for c in self.corners])

        min_distance = 10
        filtered_corners = []

        for point in grid:
            keep = Truex
            for f_point in filtered_corners:
                if np.linalg.norm(np.array(point)-np.array(f_point)) < min_distance:
                    keep = False
                    break
        
            if keep:
                filtered_corners.append(point)

        sorted_corners = sorted(filtered_corners[:324], key=lambda p: (p[1], p[0]))

        return np.array(sorted_corners)
    
    def hough(self): # Probabilistic Hough Transform for detecting the lines in the images and the filter function for filtering the lines whit respect to the corners finded in the Shi-Thomas algorithm
        lines = cv2.HoughLinesP( # Probabilistic Hough Transform
                    self.edges, # Input edge image
                    1, # Distance resolution in pixels
                    np.pi/180, # Angle resolution in radians
                    threshold=40, # Min number of votes for valid line
                    minLineLength=10, # Min allowed length of line
                    maxLineGap=20 # Max allowed gap between line for joining them
                    )
        if lines is not None:
            lines = self.filter_lines_corners(lines)

        return lines
    
    def filter_lines_corners(self, lines): # A filter function for filtering the lines whit respect to distance from the corners generated whit the shi-thomas function

        if self.corners is None or lines is None:
            return lines
        
        filtered_lines = []
        max_distance = 5

        for line in lines:
            x1, y1, x2, y2 = line[0]
            keep_line = False

            for corner in self.corners:
                cx, cy = corner.ravel()
                distance1 = np.sqrt((x1 -cx)**2 + (y1 - cy)**2)
                distance2 = np.sqrt((x2 - cx)**2 + (y2- cy)**2)

                #print(f"Distance 1  = {distance1}")
                #print(f"Distance 2  = {distance2}")

                if distance1 < max_distance or distance2 < max_distance: # LEGG IN OR
                    keep_line = True
                    break

                if keep_line:
                    filtered_lines.append(line)
    
        return filtered_lines
        #return None
    
    def extract_cells(self): # Extracting celles from the images and returning the hole matrix whit cells

        grid = np.array(self.filtered_corners) #.reshape(9, 9, 2)
        cells = np.empty((9,9), dtype= object)
            
        for i in range(9):
            for j in range(9):
                idx = i * 9 + j 
                x1, y1 = grid[idx]
                x2, y2 = grid[idx + 10]

                cell = self.image[y1 : y2, x1 : x2] # Extracting a cell from the image, self.image
                cells[i, j] = cell

        return cells # Returns the hole matrix whit sudoku cells
    
    def digit_recognize(self, cell_image):

        cell_image = cv2.resize(cell_image, (28, 28), interpolation=cv2.INTER_AREA)
        _, cell_image = cv2.threshold(cell_image, 128, 255, cv2.THRESH_BINARY_INV)

        transforms_pipeline = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        cell_image = transforms_pipeline(cell_image).unsqueeze(0)

        with torch.no_grad():
            output = self.cnn_model(cell_image)
            pred = torch.argmax(output, dim=1).item()

        return pred
    
    def number_placement(self, board):

        for i in range(9):
            for j in range(9):
                num = self.digit_recognize(self.cells[i, j])
                if board[i][j] == 0:
                    board[i][j] = num

        return board

    
    def draw_lines(self): # The lines generated whit the above function are drawed on a plane white image

        line_image = np.ones_like(self.image) * 255

        if self.lines:
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 0), 2)

        return line_image
    

path_segmented_image = "./segmented_sudoku.jpg"

cnn_model = CNN()

filter_cnn = CNN_filter("./segmented_sudoku.jpg").filtered_image()

cv2.imwrite("filtered_segmented.jpg", filter_cnn)

a = SudokuVision("filtered_segmented.jpg", cnn_model).draw_lines()

b = SudokuVision("filtered_segmented.jpg", cnn_model).shi_thomas()


plt.imshow(a)
plt.title("Drawn lines")
plt.show()


    
    
        

            
    