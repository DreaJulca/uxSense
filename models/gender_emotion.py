#TODO: Error handler for users without FFMPEG and FFProbe in path
#syntax: python models/gender_emotion.py --vid assets/[vidname].mp4
"""
Given a pose video with a start frame number and an end frame number,
generate a pose skeleton video in the GIF format

input
  video_name:   the name of a pose video
  start_frame:  the start frame number in a pose skeleton video
  end_frame:    the end frame number in the pose video

output
  a pose skeleton video file (.gif)

usage: python gen_skeleton.py --pose video_name --start start_frame_num \
          --end end_frame_num --label label_num

"""
import argparse
import os
import sys
import cv2
# GIF
import imageio

import getopt
import time
import subprocess

import json

from shutil import copyfile

# start by loading the OpenPose CNN model
model = 'mobilenet_v2_large'
w, h = 432, 368
dim = (w, h)

# list to store emotion/gender outputs
human_images = []

def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
        stdout = subprocess.PIPE, 
        stderr = subprocess.STDOUT, 
        shell=True, 
        encoding='utf8')
    return [x for x in result.stdout.readlines() if "Duration" in x]

def main(argv):
    argument = ''
    usage = 'usage: echo_args.py -f <sometext>'
    # parse incoming arguments
    try:
        opts, args = getopt.getopt(argv,"hf:",["vid="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-f", "--vid"):
            argument = arg
    # print output
    cd = os.getcwd()
    vidpath = cd + "/" + argument
    basepath = vidpath.replace(".mp4", "/")
    facepath = vidpath.replace(".mp4", "/face")

    if not os.path.exists(basepath):
        try:
            os.mkdir(basepath)
        except OSError:
            print ("Creation of the directory %s failed" % basepath)
        else:
            print ("Successfully created the directory %s " % basepath)
            
    if not os.path.exists(facepath):
        try:
            os.mkdir(facepath)
        except OSError:
            print ("Creation of the directory %s failed" % facepath)
        else:
            print ("Successfully created the directory %s " % facepath)

    # face_classification using Arriaga et al 2017
    from face_classification.src.evaluate import FaceEvaluator
    
    FaceEvaluator.load_params(cd)

    emotion_by_frame = []

    vidcap = cv2.VideoCapture(vidpath)

    success,image = vidcap.read()
    count = 0
    while success:
        # get CNN output
        gend_emot = FaceEvaluator.describe_face(image)

        gend_emot.append(count)

        #print(json.dumps(pose_coord))
        #with open(os.path.join(facepath, "frame%d.txt" % count), 'w') as f:
        #    f.write(str(json.dumps(gend_emot)))
 
        success,image = vidcap.read()
    #    #print('Read a new frame: ', success)
        emotion_by_frame.append(gend_emot)
        count += 1
    
    cv2.destroyAllWindows()

    with open(facepath + "_all_emotions_poses_gender.json", 'w') as f:
        f.write(str(json.dumps(emotion_by_frame)))



if __name__ == "__main__":
    main(sys.argv[1:])