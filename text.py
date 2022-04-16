import time

import cv2
import mss
import numpy
import pyautogui as pyautogui
import win32gui
import numpy as np
import torch

sct = mss.mss()


#     # hwnd = win32gui.FindWindow(None, 'Calculator')
# desktop = win32gui.GetDesktopWindow()
#     # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
#     # bbox = {'left': left, 'top': top, 'width': right-left, 'height': bottom-top}
# bbox = win32gui.GetWindowRect(desktop)
# screen = sct.grab(bbox)
# scr = np.array(screen)
#
#     # model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
#     # model = model.autoshape()
#     #
#     # result = model(scr)  #, size=700
#     # result.display(save=False, show=True, pprint=False)
#     # result.show()
#     cv2.imshow('window', scr)
#     cv2.waitKey(0)
#     cv2.destroyWindow()

def get_screeshot():
    screenshot = pyautogui.screenshot()
    # or: screenshot = ImageGrab.grab()
    screenshot = np.array(screenshot)
    # screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model = model.autoshape()


def run():
    while (True):
        time.sleep(0.5)
        print("frame")
        screenshot = get_screeshot()
        result = model(screenshot)  # , size=700
        # result.show()
        results = result.pandas().xyxy[0].to_dict(orient="records")
        x = numpy.array(results)
        print(x)

        # result.display(save=False, show=True, pprint=False)

        # result.show()
        # cv2.imshow('Computer Vision', screenshot)

        # press 'q' with the output window focused to exit.
        # waits 1 ms every loop to process key presses
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


run()
