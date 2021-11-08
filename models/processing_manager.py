import argparse
import os
import sys
import cv2
import getopt
import time
import subprocess
import json


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

    print('processing ' + vidpath)
   
    if not os.path.exists(vidpath.replace(".mp4", "")):
        cmd = "python models/generate_skeleton.py --vid " + argument
        returned_value = os.system(cmd)  # returns the exit code in unix
        print('returned value:', returned_value)
    else:
        print('video processing has already started')

if __name__ == "__main__":
    main(sys.argv[1:])