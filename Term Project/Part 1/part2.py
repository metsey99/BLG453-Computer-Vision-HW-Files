from cv2 import cv2
import pyautogui
import numpy as np
from utils import takeScreenShot
from collections import Counter
import time

def playPart2():
  img1 = takeScreenShot(filename="img1.png", toSave=False)
  img2 = takeScreenShot(filename="img2.png", toSave=False)
  img3 = takeScreenShot(filename="img3.png", toSave=False)
  
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
  gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

  grays = [gray1, gray2, gray3]
  guesses = []
  for _, gray in enumerate(grays):
    mask = np.uint8((gray >= 200)*255.0)
    num_labels, labels = cv2.connectedComponents(mask)

    pixel_count = []
    guess = []
    ## Append the number of classes, minimum column index and label
    for k in range(num_labels):
      x, y = np.where(labels == k)
      pixel_count.append([len(x), np.min(y), k])

    ## Sort dices from left to right with area of the side of the dice that is largest.
    pixel_count = sorted(pixel_count, key= lambda count: count[0], reverse=True)
    dices = pixel_count[1:4]
    dices = sorted(dices, key=lambda dice: dice[1])
    
    for j in range(3):
      ## Mask the largest dice side.
      label1 = np.uint8((labels == dices[j][2])*255.0)
      label1 = cv2.GaussianBlur(label1, (35,35), 2)
      corners = list(cv2.goodFeaturesToTrack(label1,4,0.001,300).reshape(-1, 2))
      
      ## If there are not 4 corners, append 0 to guesses
      if len(corners) != 4:
        guess.append(0)
        continue

      ## Preprocess corners in order to match with final points to transform
      preprocessed_corners = np.empty((4,2))
      
      top_left_corner = np.argmin(np.array(corners)[:, 0])
      preprocessed_corners[0] = corners[top_left_corner]
      corners.pop(top_left_corner)

      bot_right_corner = np.argmax(np.array(corners)[:, 0])
      preprocessed_corners[3] = corners[bot_right_corner]
      corners.pop(bot_right_corner)

      bot_left_corner = np.argmin(np.array(corners)[:, 1])
      preprocessed_corners[1] = corners[bot_left_corner]
      corners.pop(bot_left_corner)

      preprocessed_corners[2] = corners[0]
      
      ## Transform the side of the dice to a square 
      pts1 = np.float32(preprocessed_corners)
      pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
      M = cv2.getPerspectiveTransform(pts1,pts2)  
      dst = cv2.warpPerspective(gray,M,(300, 300))

      ## Detect the squares
      circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, dst.shape[0] / 4, param1=200, param2=10, minRadius=20, maxRadius=50)
      if circles is None:
        guess.append(0)
      else:  
        guess.append(circles.shape[1])

    guesses.append(guess)
  pred = np.argmax(np.array(guesses))

  if pred == 0:
    pyautogui.press('a')
  elif pred == 1:
    pyautogui.press('s')
  elif pred == 2:
    pyautogui.press('d')

if __name__ == '__main__':
  for i in range(5, 0, -1):
    time.sleep(1)
    print(i)
  while True:
    playPart2()