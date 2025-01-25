import cv2 as cv
import numpy as np
import pytesseract # extract data from webcam 

def c_vision():

    #pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\tesseract.exe'

    while True:

        phone_cam_ip = 0
        cp = cv.VideoCapture(phone_cam_ip)

        box_a, box_b, box_width, box_higth = 200, 100, 250, 300

        cam, frame = cp.read()
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)# https://stackoverflow.com/questions/42867928/why-convert-to-grayscale-opencv

        cv.rectangle(frame, (box_a, box_b), (box_a + box_width, box_b + box_higth), (0, 255, 0), 2) # https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
        
        roi = gray[box_b: box_b + box_higth, box_a: box_a + box_width] # https://answers.opencv.org/question/233775/how-can-i-modify-roi-and-then-add-it-to-the-original-image-using-python/

        thresh = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2) # https://stackoverflow.com/questions/42137362/finding-contours-from-a-thresholded-image

        contours, i = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
            if len(approx) == 4:
                cv.drawContours(frame, [approx + (box_a, box_b)], 0, (0, 255, 0), 2)

        #extract_numb = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789")
        #cleaned_numb = extract_numb.strip()

        cv.imshow("Frame", frame)
        cv.imshow("Threshold ROI", thresh)
        
        # Avslutt ved å trykke 'q'
        if cv.waitKey(1) == ord('q'):
            break

    cp.release()
    cv.destroyAllWindows()

    return frame, thresh #cleaned_numb
