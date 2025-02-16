import cv2
import numpy as np
from ultralytics import YOLO

def PaperVision(detected_sudoku):
    yolo_w = "./yolo_Weights/box_2_yolov8n.pt"

    # Last inn modellen
    model = YOLO(yolo_w)

    # Les inn bildet
    img = cv2.imread(detected_sudoku)
    if img is None:
        print(f"Feil: Kunne ikke laste bildet {detected_sudoku}")
        return None
    
    original_size = img.shape[:2]  # (høyde, bredde)
    print("Opprinnelig bildestørrelse:", original_size)

    # Konverter bilde til RGB (YOLO krever RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Kjør YOLO med segmenteringsmasker
    results = model.predict(img, imgsz=original_size, conf=0.5, agnostic_nms=True)

    # Debugging - sjekk om deteksjoner er funnet
    if not results[0].masks:
        print("Ingen segmentering funnet!")
        return None

    print("Segmenteringsmasker funnet!")

    # Hent den første segmenteringsmasken
    mask_points = results[0].masks.xy[0]  # YOLO gir en liste av segmenteringspolygoner

    # Konverter til int
    mask_points = np.array(mask_points, dtype=np.int32)

    # Lag en tom maske
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Tegn polygonet på masken
    cv2.fillPoly(mask, [mask_points], 255)

    # Bruk masken på originalbildet
    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    # Konverter tilbake til BGR for OpenCV
    segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2BGR)

    #segmented_img[:, :, 2] = mask

    # Lagre det segmenterte bildet
    cv2.imwrite("segmented_sudoku.jpg", segmented_img)
    print("Bilde lagret som segmented_sudoku.jpg")

    # Vis bildet
    cv2.imshow("Segmentert Sudoku", segmented_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return segmented_img

# Kjør funksjonen
detected_sudoku = "./Detected_sudoku_images/Detected_sudoku.jpg"
PaperVision(detected_sudoku)
