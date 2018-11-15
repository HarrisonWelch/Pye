# eye_test.py
import pyautogui
import random
import numpy as np
import cv2

# globals
mousePoint = (400,400)
points = []
lastPoint = (-1,-1)

def getLeftmostEye(eyes):
  leftMostX = 9999999
  leftMostEye = (0,0,10,10)

  for (ex,ey,ew,eh) in eyes:
    # print("ex = " + str(ex) + ", ey = " + str(ey) + ", ew = " + str(eh) + ", ey = " + str(eh) )
    if int(ex) < int(leftMostX):
      leftMostEye = (ex,ey,ew,eh)
      leftMostX = ex

  return leftMostEye

def getEyeball(frame, circles):

  # create an array of sums for every circle
  sums = np.zeros(len(circles))

  h = int(frame.shape[0] / 10)
  w = int(frame.shape[1] / 10)

  # print("getEyeball -> ")
  for y in range(0, h):
    for x in range(0, w):
      for i in range(0, len(circles)):
        # print('circles[i] = ' + str(circles[i]))
        cx = circles[i][0]
        cy = circles[i][1]
        r  = circles[i][2]
        dist_sq = ((x - cx) ** 2) + ((y - cy) ** 2)
        if dist_sq < (r ** 2):
          sums[i] = sums[i] + dist_sq

  smallestSum = 9999999
  smallestSumI = -1
  for i in range(0,i):
    if sums[i] < smallestSum:
      smallestSum = sums[i]
      smallestSumI = i

  return circles[smallestSumI]

def stabilize(points, windowsize):
  sumX = 0
  sumY = 0
  count = 0

  for i in range (0, max(0, (len(points) - windowsize))):
    sumX = sumX + points[i][0]
    sumY = sumY + points[i][1]
    count = count + 1
  
  if count > 0:
    print("count = " + str(count))
    sumX = sumX / count
    sumY = sumY / count

  return (int(sumX), int(sumY))

def moveMouse(frame, mousePoint):
  print("moving to = " + str((mousePoint[0], mousePoint[1])))
  pyautogui.moveTo(mousePoint[0], mousePoint[1], duration=0.1)

if __name__ == "__main__":
  points = []
  print("pyautogui version = " + str(pyautogui.__version__))
  print("numpy version = " + str(np.__version__))
  print("cv2 version = " + str(cv2.__version__))

  # open cam
  cap = cv2.VideoCapture(0)

  # grab the haar casscade
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

  # /haarcascade_eye_tree_eyeglasses.xml
  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Our operations on the frame come here
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = frame[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)

      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

      lme_x,lme_y,lme_w,lme_h = getLeftmostEye(eyes)

      # print("lme_x,lme_y,lme_w,lme_h = " + str((lme_x,lme_y,lme_w,lme_h)))

      leftMostEyeFrame = frame[
        y+lme_y:y+lme_y+lme_h,
        x+lme_x:x+lme_x+lme_w
      ]
      cv2.circle(frame, (lme_x+x, lme_y+y), 3, (255, 150, 150), -1)
      T = frame[70:170, 440:540]

      # cv2.cvtColor(leftMostEyeFrame, cv2.COLOR_BGR2GRAY)
      # abc = leftMostEyeFrame.astype(np.uint8)
      # abc = np.uint8(leftMostEyeFrame)
      leftMostEyeFrame_int8 = leftMostEyeFrame.astype(np.uint8)
      leftMostEyeFrame_int8 = cv2.cvtColor(leftMostEyeFrame_int8, cv2.COLOR_BGR2GRAY)
      leftMostEyeFrame_int8 = cv2.equalizeHist(leftMostEyeFrame_int8)
      # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
      # leftMostEyeFrame_int8 = clahe.apply(leftMostEyeFrame_int8)
      # leftMostEyeFrame_int8 = cv2.resize(leftMostEyeFrame_int8, (0,0), fx=8, fy=8) 

      # put down
      # image – 8-bit, single-channel, grayscale input image.
      # circles – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  (x, y, radius) .
      # circle_storage – In C function this is a memory storage that will contain the output sequence of found circles.
      # method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT , which is basically 21HT , described in [Yuen90].
      # dp – Inverse ratio of the accumulator resolution to the image resolution. 
        # For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
      # minDist – Minimum distance between the centers of the detected circles. 
        # If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
      # param1 – First method-specific parameter. 
        # In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
      # param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
      # minRadius – Minimum circle radius.
      # maxRadius – Maximum circle radius.
      circles = cv2.HoughCircles( leftMostEyeFrame_int8,
                                  cv2.HOUGH_GRADIENT,
                                  1,
                                  20,
                                  param1=5,
                                  param2=3,
                                  minRadius=20,
                                  maxRadius=40
      )
      
      # cv::HoughCircles(eye, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);

      # print("circles = " + str(circles))

      if circles is None:
        continue

      if len(circles) > 0:
      
        circles = np.uint16(np.around(circles))
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
        # print("len( circles[0,:]) = " + str(len(circles[0,:])))
        # for i in circles[0,:]:
          # draw the outer circle
          # cv2.circle(leftMostEyeFrame_int8,(i[0],i[1]),i[2],(0,255,0),2)
          # print("(i[0],i[1]),i[2] = " + str(((i[0],i[1]),i[2])))
          # draw the center of the circle
          # cv2.circle(leftMostEyeFrame_int8,(i[0],i[1]),2,(255,255,255),3)

        eye_x, eye_y, eye_r = getEyeball(leftMostEyeFrame_int8, circles[0,:])
        
        points.append( (eye_x, eye_y) )

        if len(points) > 20:
          points = points[5:]

        center = stabilize(points, 5)

        if len(points) > 1:
          diff = (((center[0] - lastPoint[0]) * 20), ((center[1] - lastPoint[1]) * -30))
          mousePoint = ( mousePoint[0] + diff[0],  
                         mousePoint[1] + diff[1] )
        lastPoint = center

        cv2.circle(leftMostEyeFrame_int8, center, 3, (0, 150, 150), -1)
        cv2.circle(leftMostEyeFrame_int8,(eye_x, eye_y),int(eye_r/3),(0,255,0),2)

        leftMostEyeFrame_int8 = cv2.resize(leftMostEyeFrame_int8, (0,0), fx=8, fy=8) 
        # print("eyeballCircle = " + str((eye_x, eye_y, eye_r)))
        cv2.imshow("leftMostEyeFrame_int8", leftMostEyeFrame_int8)

        moveMouse(frame, mousePoint)

        # cv2.equalizeHist(abc)

        # cv2.imshow('abc',abc)

    # Display Classic Frame 
    cv2.imshow('frame',frame)

    # Display Gray Frame 
    cv2.imshow('gray',gray)

    # Display the gray frame
  
    # cv2.imshow('T',T)


    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # cv2.imshow('img',img)
  # cv2.waitKey(0)
  cv2.destroyAllWindows()