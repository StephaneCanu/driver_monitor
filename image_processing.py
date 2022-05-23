import cv2
import numpy as np
from model_params import MODEL_HEIGHT, MODEL_WIDTH

W, H, BW, BH, FULL_W = 372, 864, 320*0.15, 640*0.15, 1152


def im_preprocessing(img, Is_RHD):
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


def make_attack_frame(attack_inputs, ori, rec):
    attack_inputs = attack_inputs/0.0078125 + 128.

    x1, y1 = int(rec[0] + 320 * 0.15), int(rec[1] + 640 * 0.15)
    x2, y2 = int(rec[0] + rec[2] - 320 * 0.15), int(rec[1] + rec[3] - 640 * 0.15)
    attack_crop_y = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.uint8)
    attack_crop_y[0:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2] = attack_inputs[0, 0, :, :]
    attack_crop_y[1:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2] = attack_inputs[0, 1, :, :]
    attack_crop_y[0:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2] = attack_inputs[0, 2, :, :]
    attack_crop_y[1:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2] = attack_inputs[0, 3, :, :]
    attack_crop = np.zeros(((y2-y1)*3//2, x2-x1), dtype=np.uint8)
    attack_crop_y = cv2.resize(attack_crop_y, (x2-x1, y2-y1))
    attack_crop_u = cv2.resize(attack_inputs[0, 4, :, :], ((x2-x1)//2, (y2-y1)//2))
    attack_crop_v = cv2.resize(attack_inputs[0, 5, :, :], ((x2-x1)//2, (y2-y1)//2))
    attack_crop[0:(y2-y1), :] = attack_crop_y
    attack_crop[(y2-y1):(y2-y1)*5//4, :] = attack_crop_u.reshape(-1, (x2-x1))
    attack_crop[(y2-y1)*5//4:(y2-y1)*3//2, :] = attack_crop_v.reshape(-1, (x2-x1))
    attack_crop = cv2.cvtColor(attack_crop, cv2.COLOR_YUV420p2RGB)
    frame = np.copy(ori)
    frame[y1:y2, x1:x2] = attack_crop
    return frame
