import numpy as np
import cv2

confidenceThreshold = 0.5
# Set NMS thresholds
NMSThreshold = 0.3


modelConfiguration = 'TA3/cfg/yolov3.cfg'
modelWeights = 'TA3/yolov3.weights'

labelsPath = 'TA3/coco.names'
labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

image = cv2.imread('TA3/static/img.jpg')

dimensions = image.shape[:2]
H = dimensions[0]
W = dimensions[1]


blob = cv2.dnn.blobFromImage(image, 1/255, (320,320))
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

# Apply Non Maxima Suppression to remove overlapping bounding boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

# Define the font style
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    # write condition to check the i is in indexes
    if i in indexes:
        x = boxes[i][0]
        y = boxes[i][1]
        w = boxes[i][2]
        h = boxes[i][3]

        # Get the label by the classId
        label = labels[classIds[i]]

        color = (0,0,255)

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

        # Display the confidence and label text on the image
        text = '{}: {:.2f}'.format(label, confidences[i]*100)
        cv2.putText(image, text, (x, y - 5), font, 0.5, color, 1)

cv2.imshow('Image', image)
cv2.waitKey(0)