import io

import model
import time
from flask import Flask, Response, json, request, g
import numpy as np
import cv2
import jsonpickle

app = Flask(__name__)


def responseJson(data):
    return Response(json.dumps(data), mimetype='application/json')


def get_cv_img(r):
    f = r.files['img']
    in_memory_file = io.BytesIO()
    f.save(in_memory_file)
    nparr = np.fromstring(in_memory_file.getvalue(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def getTextLineBoxes(boxes, scale=1.0):
    ret = []
    for box in boxes:
        min_x = min(int(box[0] / scale), int(box[2] / scale),
                    int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale),
                    int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale),
                    int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale),
                    int(box[5] / scale), int(box[7] / scale))

        ret.append([min_x, min_y, max_x, max_y])

    return ret


@app.before_request
def before_request():
    g.request_start_time = time.time()


@app.after_request
def after_request(res):
    print("request time: {:.03f}s".format(time.time() - g.request_start_time))
    return res


# noinspection PyTypeChecker
@app.route("/ocr", methods=['POST'])
def hello():
    img = get_cv_img(request)
    # if model == crnn ,you should install pytorch
    result, img, angle, scale = model.model(img, model='crnn', detectAngle=False)

    rois = []
    ocr_result = []

    for key in result:
        rois.append(result[key][0])
        ocr_result.append(result[key][1])

    rois = getTextLineBoxes(rois, scale)

    res = {
        "words_result_num": len(rois),
        "words_result": []
    }

    for i in range(len(rois)):
        res["words_result"].append({
            'location': rois[i],
            'words': ocr_result[i]
        })

    return responseJson(res)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
