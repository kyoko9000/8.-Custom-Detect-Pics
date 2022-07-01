# import torch
# import cv2
#
# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
#
# # Images
# img = '860x394.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# image = cv2.imread(img)
# # Inference
# results = model(img)
# # results.show()
# # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
# datas = results.xyxy[0]
# for result in datas:
#     confidence = result[4]
#     clas = int(result[5])
#     if clas == 2:
#         x1 = int(result[0])
#         y1 = int(result[1])
#         x2 = int(result[2])
#         y2 = int(result[3])
#         cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
# cv2.imshow('img', image)
# cv2.waitKey(0)
# cv2.destroyWindow()
# datas = results.pandas().xyxy[0] # show data with 'pandas'
# print(results.xyxy[0]) # show data with 'tensor'

import numpy
import torch
import cv2

# load model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'ultralytics/yolov5', 'yolov5s'
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # default
model = torch.hub.load('yolov5-master', 'custom', path='yolov5s.pt', source='local')
model.conf = 0.1
# model = torch.hub.load('path/to/yolov5', 'custom', path='yolov5-master/best.pt', source='local')  # local repo

# load picture
frame = cv2.imread('A_test.jpg')

# detect
# detections = model(frame[..., ::-1])
detections = model(frame)

# print results
results = detections.pandas().xyxy[0].to_dict(orient="records")
x = numpy.array(results)
print(x)

# filter
for result in results:
    confidence = result['confidence']
    name = result['name']
    clas = result['class']
    if clas == 0:
        x1 = int(result['xmin'])
        y1 = int(result['ymin'])
        x2 = int(result['xmax'])
        y2 = int(result['ymax'])
        print(x1, y1, x2, y2)

        # draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.rectangle(frame, (x1, y1-30), (x1+120, y1), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, name, (x1+3, y1-10), cv2.FONT_HERSHEY_DUPLEX, 1, (60, 255, 255), 1)

# show pics
cv2.imshow('img', frame)
cv2.waitKey(0)
