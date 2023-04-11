
import depthai as dai
import cv2
import time
import numpy as np

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(40)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName('rgb')
camRgb.preview.link(xoutRgb.input)

labelMap = [ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "trail", "tvmonitor"]

nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

# https://blobconverter.luxonis.com/
# Choose the mobilenet-ssd, shaves = 6
nn.setBlobPath('./models/mobilenet-ssd.blob')
nn.setConfidenceThreshold(0.5)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)
camRgb.preview.link(nn.input)

nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName('nn')
nn.out.link(nnOut.input)

nnNetworkOut = pipeline.create(dai.node.XLinkOut)
nnNetworkOut.setStreamName('nnNetwork')
nn.outNetwork.link(nnNetworkOut.input)

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def displayFrame(name, frame, detections):
    for detection in detections:
        bbox = frameNorm(
            frame,
            (detection.xmin, detection.ymin, detection.xmax, detection.ymax),
            )
        cv2.putText(
            frame,
            labelMap[detection.label],
            (bbox[0] + 10, bbox[1] + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            (0,0,255),
            )
        cv2.putText(
            frame,
            f'{int(detection.confidence*100)}%',
            (bbox[0] + 10, bbox[1] + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            (0,0,255),
            )
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 2)
    cv2.imshow(name, frame)

def print_neural_network_layer_names(inNN):
    toPrint = "Output layer names:"
    for ten in inNN.getAllLayerNames():
        print(f'{toPrint},{ten},')

with dai.Device(pipeline) as device:
    print ("Connected Device:", device.getConnectedCameras())
    print("USB Speed:", device.getUsbSpeed().name)

    qRgb = device.getOutputQueue(
            name='rgb',
            maxSize=4,
            blocking=False,
            )
    qDet = device.getOutputQueue(
            name='nn',
            maxSize=4,
            blocking=False,
            )
    qNN = device.getOutputQueue(
            name='nnNetwork',
            maxSize=4,
            blocking=False,
            )

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    printOutputLayersOnce = True

    while True:
        inRgb = qRgb.tryGet()
        inDet = qDet.tryGet()
        inNN = qNN.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(
                    frame,
                    'NN fps: {:.2f}'. format(counter/(time.monotonic() - startTime)),
                    (2, frame.shape[0] - 4),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255),
                    )

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if printOutputLayersOnce and inNN is not None:
            print_neural_network_layer_names(inNN)
            printOutputLayersOnce = False

        if frame is not None:
            displayFrame('object_detection', frame, detections)

        if cv2.waitKey(1) == ord('q'):
            break

