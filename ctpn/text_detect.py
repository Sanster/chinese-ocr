import time

from ctpn.ctpn.model import ctpn
from ctpn.ctpn.detectors import TextDetector
from ctpn.ctpn.other import draw_boxes
import numpy as np


def get_boxes(bboxes):
    """
        boxes: bounding boxes
	"""
    text_recs = np.zeros((len(bboxes), 8), np.int)
    index = 0
    for box in bboxes:
        b1 = box[6] - box[7] / 2
        b2 = box[6] + box[7] / 2
        x1 = box[0]
        y1 = box[5] * box[0] + b1
        x2 = box[2]
        y2 = box[5] * box[2] + b1
        x3 = box[0]
        y3 = box[5] * box[0] + b2
        x4 = box[2]
        y4 = box[5] * box[2] + b2

        disX = x2 - x1
        disY = y2 - y1
        width = np.sqrt(disX * disX + disY * disY)
        fTmp0 = y3 - y1
        fTmp1 = fTmp0 * disY / width
        x = np.fabs(fTmp1 * disX / width)
        y = np.fabs(fTmp1 * disY / width)
        if box[5] < 0:
            x1 -= x
            y1 += y
            x4 += x
            y4 -= y
        else:
            x2 += x
            y2 += y
            x3 -= x
            y3 -= y

        text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
        index = index + 1

    return text_recs


def text_detect(img):
    print("ctpn steps:")
    ctpn_start_time = time.time()
    scores, boxes, img, scale = ctpn(img)
    print("     ctpn: {:.03f}s".format(time.time() - ctpn_start_time))

    text_start_time = time.time()
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    text_recs = get_boxes(boxes)

    print("     text detect: {:.03f}s".format(time.time() - text_start_time))
    return text_recs, None, img, scale
