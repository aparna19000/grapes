Add a Tracker
==============
In this activity, you will learn to initialize a tracker to track the object.

<img src= "https://media.slid.es/uploads/1525749/images/10493816/balltracking.gif" width = "480" height = "320">

Follow the given steps to complete this activity:

1. ### Load and initialize the Tracker
* Open the main.py file.
* Load the tracker using `cv2.legacy.TrackerCSRT_create()` and set it to a tracker variable.

    `tracker = cv2.legacy.TrackerCSRT_create()`

* Declare a flag variable `detected` and set it to `False` to update if the basketball is detected.

    `detected = False`

* Check if the basketball is not detected using an if condition.

    `if detected == False:`

* Use `tracker.init()` to initialize the tracker.

    `tracker.init(image, boxes[i])`

* Set the `detected` flag variable to `True`.

    `detected =True`

* Draw a box on successful tracking.

* Use `tracker.update(image)` to update the tracker information.

    `trackerInfo = tracker.update(image)`

* Update `trackerInfo` in image and the bounding box.

  `success = trackerInfo[0]`

  `bbox = trackerInfo[1]`


* Draw a box on successful tracking else displaying a notification.

`if success:`

    `drawBox(image, bbox)`

`else:`

    `cv2.putText(image,"Ball not found",(75,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)`

* Save and run the code to check the output.
