import cv2
import numpy as np
vid_capture = cv2.VideoCapture('people.mp4')

if (vid_capture.isOpened() == False):
  print("Error opening the video file")
else:
  fps = vid_capture.get(5)
  print('Frames per second : ', fps,'FPS')
 
  frame_count = vid_capture.get(7)
  print('Frame count : ', frame_count)
 
while(vid_capture.isOpened()):
  ret, frame = vid_capture.read()
  if ret == True:
    cv2.imshow('Frame',frame)
    # 20 is in milliseconds, try to increase the value, say 50 and observe
    key = cv2.waitKey(20)
     
    if key == ord('q'):
      break
  else:
    break
 
#vid_capture.release()
#cv2.destroyAllWindows()
# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()

# Load COCO class names (80 classes including "person")
# List of class names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Open video file
cap = cv2.VideoCapture("C:\college\VS code\Python\people.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Information to display on the screen
    class_ids = []
    confidences = []
    boxes = []
    person_count = 0

    # Post-processing
    for out in outs:
        for detection in out:
            scores = detection[0]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # 0 corresponds to "person" class
                person_count += 1
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)

                # Rectangle coordinates
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

print("Scores shape:", scores.shape)
print("Class ID:", class_id)
print("Number of classes in your YOLO model:", num_classes)  # Replace with the actual number of classes in your model

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"Person {i + 1}"
            confidence = confidences[i]
            color = (0, 255, 200)  # Green for person

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"Persons: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break

cap.release()
cv2.destroyAllWindows()
