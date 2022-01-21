
# vídeo explicando: https://www.youtube.com/watch?v=Sx_HioMUtiY

import cv2 
import time



# Classes colors

COLORS = [(0,255,255), (255,255,0),(0,255,0),(255,0,0)]


# loading classes

class_names =[]
with open("coco.names","r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Loading videos/opening webcam
cap = cv2.VideoCapture(0)


# NeuralNet weights

net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg")

# NeuralNet parameters

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (416,416), scale=1/225)


# Video reading

while True:
    _, frame = cap.read()
    start = time.time()
    classes, scores, boxes = model.detect(frame,0.1,0.2)
    end = time.time()

    for (classid,score, box) in zip(classes,scores,boxes):
        
        color = COLORS[int(classid) % len(COLORS)]

        label =  f"{class_names[classid]} : {score}"

        cv2.rectangle(frame,box,color,2)

        cv2.putText(frame,label,(box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Detection time
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"

     # tagging image
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),3)


     # opening image
    cv2.imshow("detections", frame)


     # response wait 
    if cv2.waitKey(1) == 27:
        break

# camera release
cap.release()
cv2.destroyAllWindows()    
    








