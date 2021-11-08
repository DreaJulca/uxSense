import os
import sys
import cv2
import json

w, h = 137, 80
dim = (w, h)
fps = 15
cd = os.curdir

os.chdir(cd + '/public')
framepath = 'frames/'
vidcap = cv2.VideoCapture('TableauUser.mp4')

success,image = vidcap.read()
count = 0
while success:
    frame = cv2.resize(image, dim)
    
    if (count % fps == 0):
        cv2.imwrite(framepath + "frame%d.png" % (count/fps), frame)     
        
    success,image = vidcap.read()
#    #print('Read a new frame: ', success)
    count += 1

cv2.destroyAllWindows()
vidout.release()

