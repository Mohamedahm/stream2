from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
from ultralytics import YOLO
import math

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24") if not isinstance(frame, np.ndarray) else frame

    model = YOLO("yolo-Weights/yolov8s.pt")
     
     # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier", "toothbrush"
                   ]
     
   
        
    results = model(img, stream=True)
    for r in results:
            boxes = r.boxes
    
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
    
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)
    
                # class name
                
                cls = int(box.cls[0])
               
                print("Class name -->", classNames[cls])
                if classNames[cls] == 'person':
                # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
        
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
        
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
