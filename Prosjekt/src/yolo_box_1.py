import os
from ultralytics import YOLO
import cv2
import math


def ObjectVision():
        # Object classes
    classNames = ["Fingers", "Head", "Sudoku paper"]

        # Sett riktig filbane
    yolo_w = "./yolo_Weights/box_1_yolov8n.pt"

        # Last inn modellen
    model = YOLO(yolo_w)

        # Start webcam
    cap = cv2.VideoCapture(2) # 2 gir tilgang til mobil kamera
    cap.set(3, 640)
    cap.set(4, 480)

    bouding_box = None

    while True:

        success, img = cap.read()
        
        results = model(img, stream=True)

            # https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                rectangel = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = round(float(box.conf[0]) * 100, 2)
                cls = int(box.cls[0])

                if 0 <= cls < len(classNames): # I have put a check her to process and veryfy that the length not extend to length of the list
                    
                    
                    cv2.putText(img, f'{classNames[cls]} {confidence}%', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                else:
                    pass

                    #if confidence > 50: # scan the image whit a confidence higher then 90% and use the harris algorithem to find connections(corners) to find celles
                if confidence > 95:
                    bouding_box = img[y1:y2, x1:x2]

                else:
                    pass
                    
                        
        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return bouding_box

    # return bouding_box


sudokupaper = ObjectVision()

cv2.imwrite("Sudoku_detected_4.jpg", sudokupaper)


