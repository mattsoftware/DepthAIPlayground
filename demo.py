
import depthai as dai
import cv2
import time

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName('rgb')

camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRgb.setFps(40)
camRgb.preview.link(xoutRgb.input)

monoLeft = pipeline.create(dai.node.MonoCamera)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutLeft.setStreamName('left');
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.out.link(xoutLeft.input)

monoRight = pipeline.create(dai.node.MonoCamera)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutRight.setStreamName('right');
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.out.link(xoutRight.input)

with dai.Device(pipeline) as device:
    print ("Connected Device:", device.getConnectedCameras())
    print("USB Speed:", device.getUsbSpeed().name)

    qRgb = device.getOutputQueue(
            name='rgb',
            maxSize=4,
            blocking=False,
            )
    qLeft = device.getOutputQueue(
            name='left',
            maxSize=4,
            blocking=False,
            )
    qRight = device.getOutputQueue(
            name='right',
            maxSize=4,
            blocking=False,
            )

    while True:
        inRgb = qRgb.tryGet()
        inLeft = qLeft.tryGet()
        inRight = qRight.tryGet()

        if inRgb is not None:
            cv2.imshow('rgb', inRgb.getCvFrame())
        if inLeft is not None:
            cv2.imshow('left', inLeft.getCvFrame())
        if inRight is not None:
            cv2.imshow('right', inRight.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break

