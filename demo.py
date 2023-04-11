
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

with dai.Device(pipeline) as device:
    print ("Connected Device:", device.getConnectedCameras())
    print("USB Speed:", device.getUsbSpeed().name)

    qRgb = device.getOutputQueue(
            name='rgb',
            maxSize=4,
            blocking=False,
            )

    while True:
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            cv2.imshow('rgb', inRgb.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break

