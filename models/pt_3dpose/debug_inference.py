from detectron.infer_simple import *

import numpy as np
import json


def main(args):
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for video_name in im_list:
        out_name = os.path.join(
                args.output_dir, os.path.basename(video_name)
            )
        print('Processing {}'.format(video_name))

        clboxfile = open(out_name.replace(".mp4", "cls_boxes.json"), "r")        
        clbox = json.load(clboxfile)

        clsegfile = open(out_name.replace(".mp4", "cls_segms.json"), "r")
        clseg = json.load(clsegfile)
        
        clkeyfile = open(out_name.replace(".mp4", "cls_keyps.json"), "r")
        clkey = json.load(clkeyfile)

        clmtafile = open(out_name.replace(".mp4", "cls_metad.json"), "r")
        clmta = json.load(clmtafile)

        np.savez_compressed(out_name, boxes=clbox, segments=clseg, keypoints=clkey, metadata=clmta)
        b = np.load(out_name+'.npz')
        print(b)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
