import argparse
import os
import re
import cv2
import numpy as np
import time
from skimage.util import random_noise
import glob2
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from pathlib import Path

parser = argparse.ArgumentParser(description='Play back images from a given directory')

parser.add_argument('--indir', type=str, help='Directory containing input images.')
parser.add_argument('--outdir', type=str, help='Directory containing output images.')
parser.add_argument('--video_name', type=str, help='Name of video.')
parser.add_argument('--file_name', type=str, help='File name of input image name.')
parser.add_argument('--mode', type=str, help='day or night.', choices=["night", "day"])

args = parser.parse_args()
file = open("{}".format(args.file_name), "r")
lines = file.readlines()

    
def load_image(image_path, model=None):
    pattern = 'gbrg'
    img = Image.open(image_path)
    img = demosaic(img, pattern)
    print(img)
    return np.array(img).astype(np.uint8)


pairs = []
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('{}.mp4'.format(args.video_name),fourcc, 15.0, (1024, 256))

for idx, line in enumerate(lines):
    if idx != len(lines):
        line = line[:-1]
    filename = line.split(" ")[0]
    print(filename)
    in_image = args.indir + "/" + filename + ".png"
    out_image = args.outdir + "/" + filename + ".png"
    if not os.path.exists(in_image):
        print("in image: {} not exists".format(in_image))        
        continue
    if not os.path.exists(out_image):
        print("out image: {} not exists".format(out_image))
        continue
    in_img = cv2.imread(in_image)
    out_img = cv2.imread(out_image)
    out_img = cv2.resize(out_img, (512, 256))
    stack = np.hstack([in_img, out_img])
    print("stack: {}".format(stack.shape))
    out.write(stack)
    
out.release()