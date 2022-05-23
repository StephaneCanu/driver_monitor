import cv2
import numpy as np
import onnxruntime as ort
import torch

from attack import make_attack_img
from image_processing import make_attack_frame, im_preprocessing
from demonitoring import distracted_detect
from distraction import Is_Distraction


MODEL_HEIGHT = 640
MODEL_WIDTH = 320


def head_pose_detect(frame, is_RHD):

    # x, y, w, h = head_detect(frame, method='deep')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_y, im_u, im_v, rec = im_preprocessing(img, is_RHD)

    inputs = np.zeros((1, 6, MODEL_HEIGHT//2, MODEL_WIDTH//2), dtype=np.float32)

    inputs[0, 0, :, :] = im_y[0:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2]
    inputs[0, 1, :, :] = im_y[1:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2]
    inputs[0, 2, :, :] = im_y[0:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2]
    inputs[0, 3, :, :] = im_y[1:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2]
    inputs[0, 4, :, :] = im_u
    inputs[0, 5, :, :] = im_v

    inputs = (inputs - 128.) * 0.0078125

    ort_session = ort.InferenceSession("dmonitoring_model_version_2.onnx",
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = ort_session.run(None, {"input_img": inputs.astype(np.float32)}, )

    _, driverStats = distracted_detect(outputs)

    model_distraction = Is_Distraction(device='cpu')
    distracted = model_distraction(torch.as_tensor(outputs))
    print(distracted[0, 0].item(), driverStats.distracted)

    attack_img = make_attack_img(inputs)
    outputs_attack = ort_session.run(None, {"input_img": attack_img.astype(np.float32)}, )
    frame_attack = make_attack_frame(inputs, frame, rec)

    return outputs, outputs_attack, frame_attack, rec


