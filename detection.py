import cv2
import numpy as np
import requests
import os
import time

# === CONFIG ===
url = 'http://192.168.1.101/cam-mid.jpg'  # ESP32-CAM IP
# whT = 320
whT = 416
confThreshold = 0.25
nmsThreshold = 0.3

classesfile = 'yolo.names'
# modelConfig = 'yolov3-tiny.cfg'
# modelWeights = 'yolov3-tiny.weights'
#modelConfig = './old/yolov3.cfg'
#modelWeights = 'old/yolov3.weights'
modelConfig = 'yolov3-spp.cfg'
modelWeights = 'yolov3-spp_final.weights'
# === FILE CHECKS ===
for f in [classesfile, modelConfig, modelWeights]:
    if not os.path.isfile(f):
        print(f"File not found: {f}")
        exit()

# === LOAD CLASS NAMES ===
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# === LOAD MODEL ===
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# === OBJECT DETECTION FUNCTION ===
def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []

    FRUITS = ['Banana', 'Apple', 'Orange', 'Carrot', 'Lemon']
    FRUITS_SET = {f.lower() for f in FRUITS}  # Only detect these classes

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                label = classNames[classId].lower()
                if label not in FRUITS_SET:
                    continue  # Skip non-fruits

                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    MAX_BOXES = 100
    if len(bbox) > MAX_BOXES:
        print(f"Too many boxes ({len(bbox)}), skipping frame")
        return

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    indices = np.array(indices).flatten() if len(indices) > 0 else []

    for i in indices:
        box = bbox[i]
        x, y, w, h = box
        x = max(0, x)
        y = max(0, y)
        x2 = min(wT - 1, x + w)
        y2 = min(hT - 1, y + h)
        cv2.rectangle(im, (x, y), (x2, y2), (255, 0, 255), 2)
        label = f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%'
        cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print(classNames[classIds[i]])


# === MAIN LOOP ===
while True:
    try:
        response = requests.get(url, timeout=5, stream=True)
        data = np.asarray(bytearray(response.content), dtype=np.uint8)
        im = cv2.imdecode(data, -1)

        if im is None or im.shape[0] == 0 or im.shape[1] == 0:
            print("Failed to decode image or empty image. Skipping.")
            continue

        # Flip image horizontally and vertically (rotate 180 degrees)
        im = cv2.flip(im, -1)

        blob = cv2.dnn.blobFromImage(im, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layernames = net.getLayerNames()
        output_layers = net.getUnconnectedOutLayers()
        if isinstance(output_layers[0], np.ndarray):
            output_layers = output_layers.flatten()
        outputNames = [layernames[i - 1] for i in output_layers]

        outputs = net.forward(outputNames)
        findObject(outputs, im)

        cv2.imshow('Image', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optional: Limit frame rate a bit
        time.sleep(0.05)

    except Exception as e:
        print(f"Error: {e}")
        continue

cv2.destroyAllWindows()
