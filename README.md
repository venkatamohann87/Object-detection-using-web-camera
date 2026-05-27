# Workshop 2 - Object Detection Using Web Camera
## Name: Ashwath M
## Register number: 212223230023

## Aim
The main aim of this project is to **detect and recognize real-time objects** captured through a **web camera** using a pre-trained deep learning model (YOLO or OpenCV DNN). This helps demonstrate how computer vision techniques can be applied for live object recognition and tracking.

---

##  Requirements
To run this project, install the following dependencies:

###  Software & Libraries
- Python 3.x  
- OpenCV (`opencv-python`)  
- NumPy  
- Pre-trained YOLO model files:
  - `yolov4.cfg`
  - `yolov4.weights`
  - `coco.names`

## Algorithm

1. **Load the YOLO Model**
   - Import required libraries.
   - Load configuration (`.cfg`) and weights (`.weights`) files.
   - Read the class names from `coco.names`.

2. **Capture Video from Web Camera**
   - Initialize webcam using `cv2.VideoCapture(0)`.

3. **Process Each Frame**
   - Convert frame to blob using `cv2.dnn.blobFromImage`.
   - Pass the blob through the YOLO network.

4. **Identify and Classify Objects**
   - Extract class IDs, confidence scores, and bounding boxes.
   - Apply a confidence threshold to filter weak detections.

5. **Display Results**
   - Draw bounding boxes and labels on detected objects.
   - Show real-time output in a display window.
   - Press `q` to quit.

## Program 

```python
import cv2
import numpy as np

# Load YOLOv4 network
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load the COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Set up video capture for webcam
cap = cv2.VideoCapture(0)
img_counter=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    # Fix mirrored webcam feed (flip horizontally)
    frame = cv2.flip(frame, 1)

    height, width, channels = frame.shape

    # Prepare the image for YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get YOLO output
    outputs = net.forward(output_layers)
    
    # Initialize lists to store detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top-left corner of the box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the image
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the image with detected objects
    cv2.imshow("YOLOv4 Real-Time Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF

    # Save frame when 's' is pressed
    if key == ord('s'):
        img_counter += 1
        filename = f"frame_{img_counter}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
    # Exit the loop if 'q' is pressed
    elif key == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

# Read the image
img = cv2.imread('frame_1.jpg')
# Convert BGR (OpenCV default) to RGB for proper display in matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(img)
plt.axis('off')  
plt.show()

```
## Output:
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/c0d827aa-9260-4d84-b2bf-288985ee7a87" />


## Result

- The program successfully detects and labels real-world objects (e.g., person, car, dog, chair, etc.) in real-time using the webcam.

- Bounding boxes are drawn around detected objects with class names and confidence scores displayed.

- This demonstrates the use of deep learning–based object detection for live video applications.
