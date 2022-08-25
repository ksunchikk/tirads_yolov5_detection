
import os
import argparse
import cv2
import numpy
import tifffile as tiff
import torch

from PIL import ImageSequence, Image

parser = argparse.ArgumentParser(description='')

parser.add_argument("-s",
                    "--source_path",
                    type=str)
parser.add_argument("-o",
                    "--out_path",
                    type=str)

parser.add_argument("-m",
                    "--model_path",
                    type=str)


def detect(args):
    model = torch.hub.load('.', 'custom', path=args.model_path, source='local')
    temp = tiff.imread(args.source_path)
    x = 0
    multiPage = []
    for img in temp:
        res = model(img)
        res.save()
        if x != 0:
            tmp = Image.open("runs/detect/exp{x}/image0.jpg".format(x=x+1))
            multiPage.append(tmp)
        x += 1
    res = Image.open("runs/detect/exp/image0.jpg")
    res.save(args.out_path+'f', save_all=True, append_images=multiPage)
    for m in range(0, x):
        if m != 0:
            os.remove("runs/detect/exp{x}/image0.jpg".format(x=m+1))
            os.rmdir("runs/detect/exp{x}".format(x=m+1))
        else:
            os.remove("runs/detect/exp/image0.jpg")
            os.rmdir("runs/detect/exp")


if __name__ == "__main__":
    args = parser.parse_args()
    detect(args)
