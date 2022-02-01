#TODO: Error handler for users without FFMPEG and FFProbe in path
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

# list to store pose outputs
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
    duppath = vidpath.replace(".mp4", "/original/")
    pubpath = vidpath.replace(".mp4", "/").replace("assets/", "public/skelframes/")
    preppath = vidpath.replace(".mp4", "/3d_vidprep/")
    posepath = vidpath.replace(".mp4", "/estimates/")
    skelpath = vidpath.replace(".mp4", "/skeletons/").replace("assets/", "public/skelframes/")

    if not os.path.exists(basepath):
        try:
            os.mkdir(basepath)
        except OSError:
            print ("Creation of the directory %s failed" % basepath)
        else:
            print ("Successfully created the directory %s " % basepath)
        
    if not os.path.exists(duppath):
        try:
            os.mkdir(duppath)
            copyfile(vidpath, duppath + argument.replace("assets/", ""))
        except OSError:
            print ("Creation of the directory %s failed" % duppath)
        else:
            print ("Successfully created the directory %s " % duppath)
        
    if not os.path.exists(preppath):
        try:
            os.mkdir(preppath)
        except OSError:
            print ("Creation of the directory %s failed" %  preppath)
        else:
            print ("Successfully created the directory %s " %  preppath)

    if not os.path.exists(pubpath):
        try:
            os.mkdir(pubpath)
        except OSError:
            print ("Creation of the directory %s failed" %  pubpath)
        else:
            print ("Successfully created the directory %s " %  pubpath)
        
        
    if not os.path.exists(posepath):
        try:
            os.mkdir(posepath)
        except OSError:
            print ("Creation of the directory %s failed" % posepath)
        else:
            print ("Successfully created the directory %s " % posepath)
        
    if not os.path.exists(skelpath):
        try:
            os.mkdir(skelpath)
        except OSError:
            print ("Creation of the directory %s failed" % skelpath)
        else:
            print ("Successfully created the directory %s " % skelpath)

    #Create 3D pose data
    processNPPath = argument.replace(".mp4", "").replace('assets/', '')

    str_3dpose_vid_prep_cmd = 'python infer_video.py  --cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml --output-dir ' + preppath + ' --image-ext mp4 --wts weights/model_final.pkl ' + duppath
    str_3dpose_2ddata_cmd = 'python prepare_data_2d_custom.py -i "'+  preppath + '" -o ' + processNPPath
    str_3dpose_cmd = 'python run.py -d custom -k ' + processNPPath + ' -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject ' + argument.replace("assets/", "") + ' --viz-action custom --viz-camera 0 --viz-video "' + vidpath + '" --viz-output ' + pubpath + '3dposes.mp4 --viz-size 6 --viz-export "' + vidpath.replace(".mp4", "/pose_3d_coords") + '"'
    print("*****************************")
    print(str_3dpose_vid_prep_cmd)
    print(str_3dpose_2ddata_cmd)
    print(str_3dpose_cmd)
    print("*****************************")

    #os.chdir(cd + "/models/pt_3dpose/")
    #subprocess.check_call(str_3dpose_vid_prep_cmd, shell=True)
    #os.system(str_3dpose_vid_prep_cmd)

    #os.chdir(cd + "/models/pt_3dpose/data/")
    #subprocess.check_call(str_3dpose_2ddata_cmd, shell=True)
    #os.system(str_3dpose_2ddata_cmd)

    #os.chdir(cd + "/models/pt_3dpose/")
    #subprocess.check_call(str_3dpose_cmd, shell=True)
    #os.system(str_3dpose_cmd)

    #os.chdir(cd)

    # OpenPose
    from tf_openpose.src.estimator import TfPoseEstimator
    from tf_openpose.src.networks import get_graph_path, model_wh
    from tf_openpose.src import common
    import tf_openpose

    cnn_model = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    vidcap = cv2.VideoCapture(vidpath)
    #This line will be a problem if i don't handle the video processing via IDs
    
    #origWidth = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    #origHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    #print((origWidth, origHeight))
    
    fourcc = cv2.VideoWriter_fourcc(*'MPG4')
    vidout = cv2.VideoWriter(pubpath + "poses.mp4", fourcc, fps=25, frameSize = dim)
    #vidout = cv2.VideoWriter(vidpath.replace(".mp4", "/poses.mp4"), fourcc=fourcc, fps=25, frameSize=(origWidth, origHeight))


    success,image = vidcap.read()
    count = 0
    while success:
        frame = cv2.resize(image, dim)
        
        # get CNN output
        cnn_output = cnn_model.inference(frame)

        pose_coord = TfPoseEstimator.pose_coordinates(frame, cnn_output, imgcopy=False)

        #print(json.dumps(pose_coord))
        with open(posepath + "frame%d.txt" % count, 'w') as f:
            f.write(str(json.dumps(pose_coord)))
 
        # draw skeleton image
        #human_img = TfPoseEstimator.draw_humans(frame, cnn_output, imgcopy=False)
        human_img = TfPoseEstimator.draw_joints(frame, cnn_output, imgcopy=False)
        #cv2.imwrite(skelpath + "frame%d.png" % count, human_img)     
        # 
        # # save frame with pose estimate as JPEG file or video?
        vidout.write(human_img)
        success,image = vidcap.read()
    #    #print('Read a new frame: ', success)
    #   lets go ahead and downsample the shit out of it
    #    count += 1
        count += 29
    
    cv2.destroyAllWindows()
    vidout.release()

    cmd = "python models/segment.py --vid " + argument
    returned_value = os.system(cmd)  # returns the exit code in unix
    print('returned value:', returned_value)


    with open(argument.replace('.mp4', '.txt'), 'w') as f:
        f.write("Process start : %s \n" % time.ctime())
        f.write('Video path is: ' + vidpath + "\n")
        f.write("Video times and rates:\n") 
        try:
            for line in getLength(vidpath):
                f.write(line)
            f.write("End : %s" % time.ctime())
        except OSError:
            f.write("Error: ffprobe not found. Please download and install ffmpeg from https://ffmpeg.org/ and add '[ffmpeg install path]/bin' to environment PATH.")
if __name__ == "__main__":
    main(sys.argv[1:])