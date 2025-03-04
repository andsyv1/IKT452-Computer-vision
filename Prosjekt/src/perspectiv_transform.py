import cv2
import numpy as np
import matplotlib.pyplot as plt

class PerspectivTransform:
    def __init__(self, path_image):
        self.path_image = path_image
        self.img = cv2.imread(path_image)
        assert self.img is not None, "File could not be read."
        self.ordered_corners = None
        self.warped = None

    def img_processing(self):
        """
        Preprocess the image: Convert to grayscale, apply Gaussian blur, and thresholding.
        """
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh

    def order_points(self, pts):
        """
        Orders the four points of a quadrilateral into:
        [top-left, top-right, bottom-right, bottom-left].
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        return rect

    def find_corners(self):
        """
        Detects the largest quadrilateral in the image and extracts its four corners.
        """
        thresh = self.img_processing()

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (assumed to be the Sudoku grid)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate contour to get a quadrilateral
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Ensure we found a four-sided polygon
        if len(approx) == 4:
            corners = approx.reshape(4, 2)
            self.ordered_corners = self.order_points(corners).astype(int)
        else:
            raise ValueError("Could not find four corners of the Sudoku board.")

    def apply_perspective_transform(self):
        """
        Applies a perspective transform to straighten the Sudoku grid.
        """
        if self.ordered_corners is None:
            self.find_corners()

        # Destination points for perspective transform (fixed 300x300 output)
        dst_pts = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype="float32")

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(self.ordered_corners.astype("float32"), dst_pts)

        # Apply perspective transformation
        self.warped = cv2.warpPerspective(self.img, M, (300, 300))

    def display_results(self):
        """
        Displays the detected corners and the transformed Sudoku grid.
        """
        if self.ordered_corners is None:
            self.find_corners()
        if self.warped is None:
            self.apply_perspective_transform()

        # Draw the detected corners on the original image
        img_corners = self.img.copy()
        for corner in self.ordered_corners:
            cv2.circle(img_corners, tuple(corner), 10, (0, 0, 255), -1)  # Red dots

        # Display the detected corners and warped output
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Original image with detected corners
        axes[0].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Detected Sudoku Corners")
        axes[0].axis("off")

        # Warped Sudoku grid (straightened)
        axes[1].imshow(cv2.cvtColor(self.warped, cv2.COLOR_BGR2RGB))
        axes[1].set_title("Warped Sudoku Grid")
        axes[1].axis("off")

        plt.show()

        # Print the detected corner coordinates
        print("Detected Corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left):")
        print(self.ordered_corners)


# Example Usage:
if __name__ == "__main__":
    path_image = "./Detected_sudoku/Sudoku_detected_4.jpg"
    transformer = PerspectivTransform(path_image)
    transformer.find_corners()
    transformer.apply_perspective_transform()
    transformer.display_results()
