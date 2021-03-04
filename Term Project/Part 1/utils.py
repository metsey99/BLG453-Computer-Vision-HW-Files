from cv2 import cv2
import pyautogui
import numpy as np

def takeScreenShot(filename=None, toSave=False,):
  screen = pyautogui.screenshot()
  if toSave:
    screen.save(filename)
  return np.array(screen)

def readReady(filename):
  screen = cv2.imread(filename)
  screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
  return np.array(screen)

def cropImageByHalf(img):
  new_img = img[:img.shape[0]//2, :]
  return new_img