import cv2
import numpy as np
import onnxruntime as ort
from head_detect import head_detect


MODEL_HEIGHT = 640
MODEL_WIDTH = 320


def im_preprocessing(frame, Is_RHD, x=0):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    im_y, im_u, im_v = cv2.split(yuv)
    im_u = cv2.resize(im_u, (im_u.shape[1] // 2, im_u.shape[0] // 2))
    im_v = cv2.resize(im_v, (im_v.shape[1] // 2, im_v.shape[0] // 2))
    crop_width = 372

    if Is_RHD:
        rec = (0, 0, crop_width, im_y.shape[0])
        im_y = im_y[:, :crop_width]
        im_u = im_u[:, :crop_width//2]
        im_v = im_v[:, :crop_width//2]
        im_y = cv2.flip(im_y, 0)
        im_u = cv2.flip(im_u, 0)
        im_v = cv2.flip(im_v, 0)
    else:
        rec = (im_y.shape[1] - crop_width, 0, crop_width, im_y.shape[0])
        im_y = im_y[:, -crop_width:]
        im_u = im_u[:, -crop_width//2:]
        im_v = im_v[:, -crop_width//2:]

    resized_height = MODEL_HEIGHT
    resized_width = MODEL_WIDTH

    source_height = int(0.7 * resized_height)
    extra_height = int((resized_height-source_height)/2)
    extra_width = int((resized_width-source_height/2)/2)
    source_width = int(source_height/2)

    # im_y = cv2.resize(im_y, (source_width, source_height))
    # im_u = cv2.resize(im_u, (source_width//2, source_height//2))
    # im_v = cv2.resize(im_v, (source_width//2, source_height//2))
    #
    # input_im_y = np.zeros((resized_height, resized_width))
    # input_im_v = np.zeros((resized_height//2, resized_width//2))
    # input_im_u = np.zeros((resized_height//2, resized_width//2))
    #
    # input_im_y[extra_height:-extra_height, extra_width:-extra_width] = im_y
    # input_im_v[extra_height//2:-extra_height//2, extra_width//2:-extra_width//2] = im_v
    # input_im_u[extra_height//2:-extra_height//2, extra_width//2:-extra_width//2] = im_u

    im_y = im_y[extra_height:-extra_height, extra_width:-extra_width]
    im_u = im_u[extra_height//2:-extra_height//2, extra_width//2:-extra_width//2]
    im_v = im_v[extra_height//2:-extra_height//2, extra_width//2:-extra_width//2]
    input_im_y = cv2.resize(im_y, (resized_width, resized_height))
    input_im_v = cv2.resize(im_v, (resized_width//2, resized_height//2))
    input_im_u = cv2.resize(im_u, (resized_width//2, resized_height//2))

    return input_im_y, input_im_u, input_im_v, rec


def head_pose_detect(frame, is_RHD):

    # x, y, w, h = head_detect(frame, method='deep')
    im_y, im_u, im_v, rec = im_preprocessing(frame, is_RHD)

    input = np.zeros((1, 6, MODEL_HEIGHT//2, MODEL_WIDTH//2))

    input[0, 0, :, :] = im_y[0:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2]
    input[0, 1, :, :] = im_y[1:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2]
    input[0, 2, :, :] = im_y[0:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2]
    input[0, 3, :, :] = im_y[1:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2]
    input[0, 4, :, :] = im_u
    input[0, 5, :, :] = im_v

    input = (input - 128.) * 0.0078125

    ort_session = ort.InferenceSession("dmonitoring_model_version_2.onnx",
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    outputs = ort_session.run(None, {"input_img": input.astype(np.float32)}, )
    return outputs, rec


