import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

class SudokuGridProcessor:
    def __init__(self, path_image):
        self.path_image = path_image
        self.image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        assert self.image is not None, "File could not be read."

        self.cells = []
        self.fused_hor_lines = []
        self.fused_ver_lines = []
        self.intersection_points = []

    def detect_grid_points(self):
        """Extracts key grid points using feature tracking."""
        edges = cv2.Canny(self.image, 50, 150)  # Apply Canny Edge Detection
        corners = cv2.goodFeaturesToTrack(edges, maxCorners=5000, qualityLevel=0.001, minDistance=15)
        return np.float32([i.ravel() for i in corners])

    def find_lines_kmeans(self, points, n_clusters=10, axis=1):
        """Finds straight lines using KMeans clustering (without 10-bin method)."""
        lines = []
        coord_all = points[:, axis].reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans.fit(coord_all)
        cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

        for center in cluster_centers:
            mask = np.isclose(points[:, axis], center, atol=5)
            cluster_coords = points[mask]

            if len(cluster_coords) < 2:
                continue

            X = cluster_coords[:, 1 - axis].reshape(-1, 1)
            Y = cluster_coords[:, axis]
            model = LinearRegression()
            model.fit(X, Y)

            x_min, x_max = np.min(X), np.max(X)
            y_min, y_max = model.predict([[x_min]])[0], model.predict([[x_max]])[0]

            if axis == 1:
                lines.append(((x_min, y_min), (x_max, y_max)))  # Horizontal
            else:
                lines.append(((y_min, x_min), (y_max, x_max)))  # Vertical

        return lines

    def sort_and_filter_lines(self, lines, axis, threshold=10):
        """Sorts and filters lines to remove duplicates and ensure a correct grid structure."""
        lines.sort(key=lambda line: (line[0][axis] + line[1][axis]) / 2)
        filtered_lines = []
        last_value = -float("inf")

        for line in lines:
            mid_value = (line[0][axis] + line[1][axis]) / 2
            if abs(mid_value - last_value) > threshold:
                filtered_lines.append(line)
                last_value = mid_value

        while len(filtered_lines) > 10:  # Keep exactly 10 lines (9 cells + 1 border)
            filtered_lines.pop(-1)

        return filtered_lines

    def find_sudoku_lines(self):
        """Finds and filters horizontal & vertical lines using only KMeans clustering."""
        points = self.detect_grid_points()

        hor_lines_kmeans = self.find_lines_kmeans(points, axis=1)
        ver_lines_kmeans = self.find_lines_kmeans(points, axis=0)

        self.fused_hor_lines = self.sort_and_filter_lines(hor_lines_kmeans, axis=1)
        self.fused_ver_lines = self.sort_and_filter_lines(ver_lines_kmeans, axis=0)

    def find_intersection_points(self):
        """Finds intersection points of detected grid lines."""
        def line_intersection(line1, line2):
            (x1, y1), (x2, y2) = line1
            (x3, y3), (x4, y4) = line2

            denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denominator == 0:
                return None

            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
            return (px, py)

        self.intersection_points = []
        for h_line in self.fused_hor_lines:
            row_points = [line_intersection(h_line, v_line) for v_line in self.fused_ver_lines]
            self.intersection_points.append(row_points)

    def extract_cells(self):
        """Extracts individual Sudoku cells using detected grid intersections."""
        for i in range(len(self.intersection_points) - 1):
            for j in range(len(self.intersection_points[0]) - 1):
                pts = [self.intersection_points[i][j], self.intersection_points[i][j+1],
                       self.intersection_points[i+1][j], self.intersection_points[i+1][j+1]]

                xs = [pt[0] for pt in pts if pt is not None]
                ys = [pt[1] for pt in pts if pt is not None]
                if not xs or not ys:
                    continue

                min_x, max_x = int(min(xs)), int(max(xs))
                min_y, max_y = int(min(ys)), int(max(ys))
                cell_img = self.image[min_y:max_y, min_x:max_x].copy()
                self.cells.append(cell_img)

    def display_results(self):
        """Displays detected grid lines and intersections."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.image, cmap='gray')

        for (x1, y1), (x2, y2) in self.fused_hor_lines:
            ax.plot([x1, x2], [y1, y2], color='blue', linestyle='solid', linewidth=1)
        for (x1, y1), (x2, y2) in self.fused_ver_lines:
            ax.plot([x1, x2], [y1, y2], color='blue', linestyle='solid', linewidth=1)

        ax.set_title("Detected Sudoku Grid")
        plt.show()


# Example Usage:
if __name__ == "__main__":
    path_image = "./captured_images_con120/image_3_000.jpg"
    
    processor = SudokuGridProcessor(path_image)
    processor.find_sudoku_lines()
    processor.find_intersection_points()
    processor.extract_cells()
    processor.display_results()
