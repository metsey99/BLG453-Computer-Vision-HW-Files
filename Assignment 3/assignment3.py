## Author: Metehan Seyran
## Student ID: 15010903

from cv2 import cv2
import pyautogui
import time
import numpy as np
import argparse

def takeScreenShot():
  screen = pyautogui.screenshot()
  return screen

def readReady(filename):
  screen = cv2.imread("./"+filename+".png")
  return screen

def sobelFilter(img):
  sobel_filter_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_filter_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
  rows, cols = img.shape
  new_pic1 = np.zeros(img.shape)
  new_pic2 = np.zeros(img.shape)

  for row in range(rows-2):
    for col in range(cols-2):
      block = img[row:row+3, col:col+3]
      block1 = np.multiply(block, sobel_filter_vertical)
      block2 = np.multiply(block, sobel_filter_horizontal)
      new_pic1[row+1, col+1] = np.sum(block1)
      new_pic2[row+1, col+1] = np.sum(block2)
  new_pic = np.uint8(np.sqrt(new_pic1**2 + new_pic2**2))
  return new_pic

def CannyFilter(img):
  edged = cv2.Canny(img, 30, 200)
  return edged

def MinEigCornerDetector(img, window_size=3, threshold=25):
  dx, dy = np.gradient(img)

  I_xx = dx*dx
  I_xy = dx*dy
  I_yy = dy*dy

  height, width = img.shape
  corner_list = []
  out_img = np.stack((img, )*3, axis=-1)

  for row in range(window_size//2, height - window_size//2):
    for col in range(window_size//2, width - window_size//2):

      G_xx = I_xx[row - (window_size//2) : row + (window_size//2), col - (window_size//2) : col + (window_size//2)]
      G_xy = I_xy[row - (window_size//2) : row + (window_size//2), col - (window_size//2) : col + (window_size//2)]
      G_yy = I_yy[row - (window_size//2) : row + (window_size//2), col - (window_size//2) : col + (window_size//2)]

      sum_xx = np.sum(G_xx)
      sum_xy = np.sum(G_xy)
      sum_yy = np.sum(G_yy)

      G = np.array([[sum_xx, sum_xy], [sum_xy, sum_yy]])
      eigenvals = np.linalg.eigvals(G)
      min_eigenval = np.min(eigenvals)

      if min_eigenval > threshold:
        corner_list.append([row, col])
        out_img[row-3:row+3, col-3:col+3] = (0, 0, 255)
  return out_img

def cropSection(screen):
  screen = screen[int(screen.shape[0]*0.75):, int(screen.shape[1]*0.50):int(screen.shape[1]*0.75), :]
  return np.uint8(screen)

def playGame():
  ## Loop forever
  while True:
    screen = np.array(takeScreenShot())[:,:,[2,1,0]]
    
    ## Crop the part of the bottom where the shapes pass 
    subscreen = cropSection(screen)
    subscreen = cv2.cvtColor(subscreen, cv2.COLOR_BGR2GRAY)
    subscreen[np.where(subscreen == 210)] = 255

    ## Run corner detection algorithm which returns button to press
    button = cornerDetectionReady(subscreen)
    if button != -1:
      pyautogui.press(button)

def cornerDetectionReady(img):
  contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  contours.reverse()
  for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    if len(approx) == 3:
      print("Triangle")
      return "a"
    elif len(approx) == 4:
      print("Square")
      return "s"
    elif len(approx) == 6:
      print("Hexagon")
      return "f"
    elif len(approx) == 10:
      print("star")
      return "d"
    else:
      return -1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--part', type=int, required=True, help='Select the part of the HW')

  args = parser.parse_args()
  
  print("Please open the all shapes page in 5 seconds.")
  for i in range(5):
    print(5-i)
    time.sleep(1)
  
  if args.part == 1:
    
    # screen = np.array(takeScreenShot())
    screen = np.array(readReady("allShapes"))
    screen = np.mean(cv2.GaussianBlur(screen, (5,5), cv2.BORDER_DEFAULT), axis=-1)
    result = sobelFilter(screen)

    cv2.imshow("Sobel filter Result", result)
    cv2.imwrite("./sobel_filter.png", result)
    cv2.waitKey(0)

  elif args.part == 2:
  
    # screen = np.array(takeScreenShot())
    screen = np.array(readReady("allShapes"))
    screen = np.uint8(np.mean(cv2.GaussianBlur(screen, (5,5), cv2.BORDER_DEFAULT), axis=-1))
    result = CannyFilter(screen)
  
    cv2.imshow('Canny Edges', result) 
    cv2.imwrite("./canny_filter.png", result)
    cv2.waitKey(0)
  
  elif args.part == 3:

    # screen = np.array(takeScreenShot())
    screen = np.array(readReady("allShapes"))
    screen = np.mean(cv2.GaussianBlur(screen, (5,5), cv2.BORDER_DEFAULT), axis=-1)
    result = MinEigCornerDetector(screen)

    cv2.imshow("Minimum Eigenvalue Detector", result)
    cv2.imwrite("./min_eig_corner.png", result)
    cv2.waitKey(0)
  
  elif args.part == 4:

    playGame()

  else:
    print("The --part flag accepts values ranging 1 to 4 included.")