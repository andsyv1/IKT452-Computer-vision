import cv2
import numpy as np
from yolo_box_1 import ObjectVision
from perspectiv_transform import PerspectivTransform
from sudokugridprosessor import SudokuGridProcessor
from ultralytics import YOLO

# Laster inn klassifiseringsmodellen
path_classify = "./yolo_Weights/classification_yolov8n.pt"
classification_model = YOLO(path_classify)

def classify_cells(cells):
    classified_digits = []
    for cell in cells:
        if len(cell.shape) == 2:
            cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
        results = classification_model(cell)
        if results and len(results) > 0:
            top_prediction = results[0]
            class_id = int(top_prediction.probs.top1)
            classified_digits.append(class_id)
        else:
            classified_digits.append(0)
    return classified_digits

def format_sudoku_grid(classified_digits):
    if len(classified_digits) != 81:
        raise ValueError("Feil antall klassifiserte celler. Forventet 81.")
    return np.array(classified_digits).reshape(9, 9)

def is_valid_move(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    box_x, box_y = (row // 3) * 3, (col // 3) * 3
    for i in range(3):
        for j in range(3):
            if board[box_x + i][box_y + j] == num:
                return False
    return True

def solve_sudoku(board):
    empty_cell = [(r, c) for r in range(9) for c in range(9) if board[r][c] == 0]
    
    def backtrack(index=0):
        if index == len(empty_cell):
            return True
        row, col = empty_cell[index]
        for num in range(1, 10):
            if is_valid_move(board, row, col, num):
                board[row][col] = num
                if backtrack(index + 1):
                    return True
                board[row][col] = 0
        return False
    
    backtrack()
    return board

def plot_sudoku_solution(image_path, solved_board):
    image = cv2.imread(image_path)
    if image is None:
        print("Kunne ikke laste bildet.")
        return
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cell_size = image.shape[0] // 9
    
    for i in range(9):
        for j in range(9):
            num = solved_board[i][j]
            if num != 0:
                x, y = j * cell_size + cell_size // 3, i * cell_size + cell_size // 1.5
                cv2.putText(image, str(num), (x, y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Løst Sudoku", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Kjører YOLO-deteksjon...")
    sudoku_image = ObjectVision()
    if sudoku_image is None:
        print("Feil: Sudoku-brett ikke funnet.")
        return
    
    image_path = "sudoku_detected.jpg"
    cv2.imwrite(image_path, sudoku_image)
    
    print("Retter opp perspektivet...")
    transformer = PerspectivTransform(image_path)
    transformer.find_corners()
    transformer.apply_perspective_transform()
    transformed_image_path = "sudoku_warped.jpg"
    cv2.imwrite(transformed_image_path, transformer.warped)
    
    print("Prosesserer Sudoku-rutenett...")
    processor = SudokuGridProcessor(transformed_image_path)
    processor.find_sudoku_lines()
    processor.find_intersection_points()
    processor.extract_cells()
    
    print("Klassifiserer tall i Sudoku-rutenettet...")
    classified_digits = classify_cells(processor.cells)
    sudoku_grid = format_sudoku_grid(classified_digits)
    
    print("Løst Sudoku-rutenett:")
    solved_board = solve_sudoku(sudoku_grid)
    print(np.array(solved_board))
    
    plot_sudoku_solution(transformed_image_path, solved_board)
    
if __name__ == "__main__":
    main()
