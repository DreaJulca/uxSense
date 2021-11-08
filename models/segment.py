#TODO: Error handler for users without FFMPEG and FFProbe in path
"""
Stacks all the joint coordinates in a table, then 
    calls R implementation of e-divisive.

2D skeletons have joints numbered 0-17 
    (per https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/pose_dataset.py)

3D skeletons have joints numbered 
    i don't know yet because i haven't implemented it at this point 
    and will need to modify this with additional features when i do 
"""
import argparse
import os
import sys
import cv2
# GIF
import imageio

import math 

import getopt
import time
import subprocess

import json
import csv

w, h = 432, 368
dim = (w, h)

def main(argv):
    argument = ''
    usage = 'usage: echo_args.py -f <sometext>'
    # parse incoming arguments
    is3d = False

    try:
        opts, args = getopt.getopt(argv,"hf:",["vid=", "3d="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        if opt in ("-f", "--vid"):
            argument = arg
        if opt in ("--3d"):
            is3dStr = arg
            if is3dStr[0:1].lower() in ('t', 'y'):
                is3d = True

    # print output
    cd = os.getcwd()
    vidpath = cd + "/" + argument

    if os.path.exists(vidpath.replace(".mp4", "/3d_pose_coordinates.json")):
        outfile = vidpath.replace(".mp4", "/tf-only_2d_all_pose_estimates.csv")
    else:
        outfile = vidpath.replace(".mp4", "/all_pose_estimates.csv")
    
    posepath = vidpath.replace(".mp4", "/estimates/")
    framefiles = [f for f in os.listdir(posepath) if os.path.isfile(os.path.join(posepath, f))]
    with open(outfile, 'w', newline='') as outf:
        datastream = csv.writer(outf)
        headers = []
        for i in range(18):
            for j in range(3):
                if j % 3 == 0:
                    coordplane = "x"
                if j % 3 == 1:
                    coordplane = "y"
                if j % 3 == 2:
                    coordplane = "z"

                headers.append(coordplane + "_" + str(i))

        datastream.writerow(headers)

        for f in framefiles:
            with open(os.path.join(posepath, f)) as json_file:
                data = json.load(json_file)
                row = []
                for i in range(18):
                    try:
                        for x in data[str(i)]:
                            row.append(x)
                    except:
                        #print("no value for joint " + str(i))
                        for x in range(2):
                            row.append(0)
                    if is3d in (0, "0", "no", "false"):
                        row.append(0)

                datastream.writerow(row)

    segfile = vidpath.replace(".mp4", "/frame_segments.json")
    poselabels = "models/2d_skeleton_labels.json"

    if os.path.exists(vidpath.replace(".mp4", "/3d_pose_coordinates.json")):
        poselabels = "models/3d_skeleton_labels.json"
        outfile = vidpath.replace(".mp4", "/all_pose_estimates.csv")
    with open(outfile, 'w', newline='') as outf:
        datastream = csv.writer(outf)
        headers = []
        for i in range(17):
            for j in range(3):
                if j % 3 == 0:
                    coordplane = "x"
                if j % 3 == 1:
                    coordplane = "y"
                if j % 3 == 2:
                    coordplane = "z"

                headers.append(coordplane + "_" + str(i))

        datastream.writerow(headers)

        with open(vidpath.replace(".mp4", "/3d_pose_coordinates.json")) as json_file:
            data = json.load(json_file)
            for frame in data:
                row = []
                for i in range(17):
                    row.append(frame[i][0])
                    row.append(frame[i][1])
                    row.append(frame[i][2])
                    
                datastream.writerow(row)

    cmd = "Rscript models/segment.R " + outfile.replace(cd, '') + " " + segfile + " " + poselabels
    print(cmd)
    returned_value = os.system(cmd)  # returns the exit code in unix
    print('returned value:', returned_value)


if __name__ == "__main__":
    main(sys.argv[1:])
