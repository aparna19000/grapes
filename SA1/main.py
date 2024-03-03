import numpy as np
import cv2


# Load the tracker


confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

video = cv2.VideoCapture("bb2.mp4")

# Flag variable to show whether basketball is detected or not



def drawBox(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    # Draw rectangle and display a notification Tracking


while True:
    check, image = video.read()
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    dimensions = image.shape[:2]
    H, W = dimensions

    # Detect the ball only if not detected earlier
    
    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
    yoloNetwork.setInput(blob)

    layerName = yoloNetwork.getUnconnectedOutLayersNames()
    layerOutputs = yoloNetwork.forward(layerName)

    boxes = []
    confidences = []
    classIds = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIds.append(classId)

    detectionNMS = cv2.dnn.NMSBoxes(
        boxes, confidences, confidenceThreshold, NMSThreshold)

    if (len(detectionNMS) > 0):
        for i in detectionNMS.flatten():

            if labels[classIds[i]] == "sports ball":
                x, y, w, h = boxes[i]

                color = (255, 0, 0)

                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                # Initialise the tracker on the img and the bounding box
                
                # Changing flag variable value to true
                
    
        # Update the tracker on the img and the bounding box
       

        # Call drawBox() if the tracking is successful
        

    cv2.imshow('Image', image)
    cv2.waitKey(1)

    key = cv2.waitKey(25)
    if key == 32:
        print("Stopped")
        break
