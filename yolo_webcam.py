from ultralytics import YOLO 
import cv2 
import cvzone

# RUNNING YOLO WITH A WEBCAM
model = YOLO("yolo_weights/yolo11n.pt")  # load a pretrained YOLOv
cap = cv2.VideoCapture(0)

cap.set(3, 720) # width is identified by the id "3", then we specify its value "1280"
cap.set(4, 480) # height is identified by the id "4", then we specify its value "720" Those are cv conventions !!

while True:

    success, img = cap.read()
    cv2.imshow("Image", img) #Show capture view, window name is "Image" and i takes it's view from img var which reads read() from cap -> webcam
    cv2.waitKey(1)  # Wait 1ms 