from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
from itertools import combinations
import math

def calculate_distance(dx, dy, dz):
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distance

# Tiny yolo v3/4 label texts
labelMap = [
    "person"
]

syncNN = True

# Get argument first
nnBlobPath = str(Path('models/tiny-yolo-v4_openvino_2021.2_6shave.blob').resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")


colorCam.setPreviewSize(416, 416)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(255)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)
# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# Create outputs

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    while True:
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame()

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        detections = inNN.detections

        if len(detections) != 0:

        # If the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width  = frame.shape[1]
            centroid_dict = dict()
            objectId = 0
            for detection in detections:
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                if str(label) == 'person' :
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                    # Denormalize bounding box
                        x1 = int(detection.xmin * width)
                        x2 = int(detection.xmax * width)
                        y1 = int(detection.ymin * height)
                        y2 = int(detection.ymax * height)
                        center = (int((x1+x2)/2), int((y1+y2)/2))
                        xsp, ysp, zsp = int(detection.spatialCoordinates.x), int(detection.spatialCoordinates.y), int(detection.spatialCoordinates.z)
                        centroid_dict[objectId] = (x1, x2, y1, y2, xsp, ysp, zsp, center)
                        objectId += 1

            red_zone_list = [] # List containing which Object id is in under threshold distance condition.
            red_line_list = []
            dist_list = []
            for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
                dx, dy, dz = p1[4] - p2[4], p1[5] - p2[5], p1[6] - p2[6]
                distance = calculate_distance(dx, dy, dz)
                if(int(distance) < 1000 and int(distance) != 0):
                    start_point=p1[7]
                    end_point=p2[7]
                    text_coord = (int((start_point[0]+end_point[0])/2),int((start_point[1]+end_point[1])/2)+20)
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
                    cv2.putText(frame, str(round(distance/500,2))+' ft', text_coord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)

            for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
                if idx in red_zone_list:   # if id is in red zone list
                    cv2.rectangle(frame, (box[0], box[2]), (box[1], box[3]), (0, 0, 255), 2) # Create Red bounding boxes  #starting point, ending point size of 2
                else:
                    cv2.rectangle(frame, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 2)

            text = "No of at-risk people: %s" % str(int(len(red_zone_list)/2)) 			# Count People at Risk
            location = (10,25)												# Set the location of the displayed text
            cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
