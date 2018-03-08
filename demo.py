# coding:utf-8
import model
from glob import glob
import numpy as np
from PIL import Image
import time

from api import getTextLineBoxes

paths = glob('./test/*.*')

if __name__ == '__main__':
    im = Image.open(paths[0])
    img = np.array(im.convert('RGB'))
    t = time.time()
    # if model == crnn ,you should install pytorch
    result, img, angle,scale = model.model(img, model='crnn', detectAngle=False)
    print("It takes time:{}s".format(time.time() - t))
    print("---------------------------------------")
    print("图像的文字朝向为:{}度\n".format(angle), "识别结果:\n")

    for key in result:
        print("[{}] {}".format(result[key][0], result[key][1]))

    rois = []

    for key in result:
        rois.append(result[key][0])

    rois = getTextLineBoxes(rois, scale)

