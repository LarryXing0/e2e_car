# get onnx output
import onnx
import onnxruntime as rt
import numpy as np
import time
import torch
import torch.nn.functional as F
import cv2
from demo.det.utils import add_bbox_on_img, box_cxcywh_to_xyxy, load_img, Timer

img_h, img_w = 750, 800
onnx_file = "/home/xy/work/demo/detr/worker_dirs/deter.onnx"
img_path = "demo/data/persons.jpg"
# load onnx graph
onnx_model = onnx.load(onnx_file)
onnx.checker.check_model(onnx_model)
# fetch input meta
input_all = [node.name for node in onnx_model.graph.input]
input_initializer = [node.name for node in onnx_model.graph.initializer]
net_feed_input = list(set(input_all) - set(input_initializer))
assert len(net_feed_input) == 1


# create onnxruntime inference session
sess = rt.InferenceSession(onnx_file, providers=["CUDAExecutionProvider"])
print(rt.get_device())

# run inference
fake_input = load_img(img_path, (img_h, img_w))
timer = Timer()
timer.start()
onnx_result = sess.run(None, {net_feed_input[0]: fake_input / 255.0 - 1.0})
timer.end()
out_logits = torch.tensor((onnx_result[0][0]))
out_bbox = onnx_result[1][0]
prob = F.softmax(torch.tensor(out_logits), -1)
scores, labels = prob[..., :-1].max(-1)

boxes = box_cxcywh_to_xyxy(torch.tensor(out_bbox))
# and from relative [0, 1] to absolute [0, height] coordinates
img_h, img_w = torch.tensor([img_h]), torch.tensor([img_w])
scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
boxes = boxes * scale_fct[:, None, :]
results = [
    {"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes[0])
]
img = np.transpose(fake_input[0], (1, 2, 0))
for res in results:
    img = add_bbox_on_img(img, res)
cv2.imwrite("worker_dirs/deter_det.jpg", img)
