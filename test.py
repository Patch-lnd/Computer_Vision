from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

result = model("images/img2.jpg", show=True)
cv2.waitKey(0)  