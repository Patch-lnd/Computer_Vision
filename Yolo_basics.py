#RUNNING YOLO WITH A WEBCAM
from ultralytics import YOLO 
import cv2
import cvzone 
import math 

capture = cv2.VideoCapture(0)  # capture video from the webcam (0 is usually the default camera)
capture.set(3, 940)  # set width of the video frame
capture.set(4, 640)  # set height of the video frame

model = YOLO("yolo_weights/yolo11n.pt")

while True:
    success, img = capture.read()  # read a frame from the webcam

    results = model(img, stream=True)  # predict on the frame

    for r in results:
        boxes= r.boxes
        for box in boxes:
            #Opencv Syntax for Rectangles
            
            x1,y1,x2,y2 = box.xyxy[0]  # get the coordinates of the bounding box, getting the first element.
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)  # convert to integers
           # cv2.rectangle(img,(x1,y1), (x2,y2),(255,0,255),3)  # draw a rectangle around the detected object 

            conf = math.ceil((box.conf[0]*100))/100 # get the confidence score of the detection
            print(conf)  # print the confidence score to the console

            cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35, y1))) #To make sure that the confidence score is not displayed outside the webcam window, we use max() to make sure that the coordinates are not negative.
            #cvzone.putTextRect(img,f'{conf}',(x1,y1-20)) #Same thing without max() but the confidence score can be displayed outside the webcam window if the object is detected at the top of the window. 


            
            #cvzone Syntax for Rectangles
            
            w, h = x2-x1, y2-y1  # get the coordinates of the bounding box in xywh format
            cvzone.cornerRect(img, (x1, y1, w, h))
            

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)  # wait for a key press to close the webcam window


 






# RUNNING YOLO WITH AN IMAGE/FILE IN COMPUTER
""" 
from ultralytics import YOLO 
import cv2

model = YOLO("yolo11n.pt")  # load a pretrained YOLOv11n model (recommended for training)

results = model("images/img1.jpg")  # predict on an image

annotated_frame = results[0].plot()
# Notre propre fenêtre OpenCV, qu'on contrôle nous-mêmes
cv2.imshow("Résultat détection", annotated_frame)


cv2.waitKey(0)  # wait for a key press to close the image window
cv2.destroyAllWindows() 

 """