import os
from ultralytics import YOLO
import cv2

def ObjectVision():
    # Object classes
    classNames = ["Fingers", "Head", "Sudoku paper"]

    # Sett riktig filbane
    yolo_w = "yolo_Weights/box_1_yolov8n.pt"

    # Last inn modellen
    model = YOLO(yolo_w)

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Opprett mappe for lagrede bilder
    save_path = "captured_images_con120"
    os.makedirs(save_path, exist_ok=True)

    image_count = 0  # Teller hvor mange bilder som er lagret

    while image_count < 50:  # Lagre maks 100 bilder
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(float(box.conf[0]) * 100, 2)
                cls = int(box.cls[0])

                if 0 <= cls < len(classNames):  # Sjekk at cls er gyldig indeks
                    cv2.putText(img, f'{classNames[cls]} {confidence}%', (x1, y1), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Hvis confidence er over 90/80%, lagre bildet
                if confidence > 90:
                    bounding_box = img[y1:y2, x1:x2]
                    if bounding_box.size != 0:
                        filename = os.path.join(save_path, f"image_3_{image_count:03d}.jpg")
                        cv2.imwrite(filename, bounding_box)
                        print(f"Lagret: {filename}")
                        image_count += 1  # Øk telleren

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break  # Avslutt hvis 'q' trykkes

    cap.release()
    cv2.destroyAllWindows()
    print("Ferdig! 100 bilder er lagret.")

    return None

ObjectVision()