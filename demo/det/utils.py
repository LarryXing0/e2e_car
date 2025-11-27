import numpy as np
import cv2
import torch
import time


class Timer:
    def __init__(self, info="Timer"):
        self.tag = info

    def start(self):
        self.curr_time = time.perf_counter()

    def end(self):
        print(self.tag, ": ", time.perf_counter() - self.curr_time)


def add_bbox_on_img(img, res, conf_thr=0.95, color=(255, 0, 0), thickness=3):
    box = res["boxes"]
    if res["scores"] > conf_thr:
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            thickness,
        )
    return img


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def load_img(img_path, input_size=(750, 800), transpose=True):
    if len(input_size) == 2:
        input_size = (input_size[0], input_size[1], 3)
    img = cv2.imread(img_path)
    raw_img = np.ones(input_size)
    width_ratio, height_ratio = (
        img.shape[0] / input_size[0],
        img.shape[1] / input_size[1],
    )
    if max(width_ratio, height_ratio) < 1.0:
        raw_img[0 : img.shape[0], 0 : img.shape[1], :] = img
    else:
        max_ratio = max(width_ratio, height_ratio)
        resized_img = cv2.resize(img, None, fx=1.0 / max_ratio, fy=1.0 / max_ratio)
        raw_img[0 : resized_img.shape[0], 0 : resized_img.shape[1], :] = resized_img
    if transpose:
        raw_img = (
            torch.tensor(
                np.array(
                    np.expand_dims(np.transpose(raw_img, (2, 0, 1)), axis=0),
                    dtype=float,
                )
            )
            .to(torch.float)
            .numpy()
        )
    return raw_img
