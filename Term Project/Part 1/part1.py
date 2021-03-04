from cv2 import cv2
import pyautogui
import numpy as np
from utils import takeScreenShot, readReady, cropImageByHalf
from collections import Counter
import time

def playPart1():
  img = takeScreenShot(filename="part1.png", toSave=False)
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  mask = np.uint8((gray >= 200)*255.0)
  _, labels = cv2.connectedComponents(mask)

  dices = []
  for i in range(1,4):
    dice_mask = np.uint8((labels == i) * 255.)
    dice_mask = cv2.GaussianBlur(dice_mask, (35,35), 1)
    
    corners = np.array(cv2.goodFeaturesToTrack(dice_mask,4,0.001,300).reshape(-1, 2))
    
    tl_y = int(np.min(corners[:,0]))
    tl_x = int(np.min(corners[:,1]))
    br_y = int(np.max(corners[:,0]))
    br_x = int(np.max(corners[:,1]))

    dice = np.uint8(gray[tl_x:br_x, tl_y:br_y])

    dices.append(dice)

  ## Based on intersecting lines, detect dice corners and crop them
  circle_count = []
  for i, dice in enumerate(dices):
    circles = cv2.HoughCircles(dice, cv2.HOUGH_GRADIENT, 1, dice.shape[0] / 4, param1=200, param2=10, minRadius=20, maxRadius=50)
    if circles is not None:
      circle_count.append(circles.shape[1])
    else:
      circle_count.append(0)  
  ind = np.argmax(circle_count)

  if ind == 0:
    pyautogui.press('a')
  elif ind == 1:
    pyautogui.press('s')
  elif ind == 2:
    pyautogui.press('d')
    

if __name__ == '__main__':
  for i in range(5, 0, -1):
    time.sleep(1)
    print(i)
  while True:
    playPart1()