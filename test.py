from dataset_pandora import PandoraData
from image_processing import im_preprocessing
import onnxruntime as ort
import cv2
import torch
from model_params import MODEL_WIDTH, MODEL_HEIGHT
import numpy as np
from demonitoring import distracted_detect
from attack import make_attack_img
from camera_param import trans_webcam_to_eon_front_pandora, WEBCAM_WIDTH, WEBCAM_HEIGHT
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def im_preprocessing(img, Is_RHD):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    im_y, im_u, im_v = cv2.split(yuv)
    im_u = cv2.resize(im_u, (im_u.shape[1] // 2, im_u.shape[0] // 2))
    im_v = cv2.resize(im_v, (im_v.shape[1] // 2, im_v.shape[0] // 2))
    crop_width = 372

    if Is_RHD:
        rec = (0, 0, crop_width, im_y.shape[0])
        im_y = im_y[:, :crop_width]
        im_u = im_u[:, :crop_width // 2]
        im_v = im_v[:, :crop_width // 2]
        im_y = cv2.flip(im_y, 0)
        im_u = cv2.flip(im_u, 0)
        im_v = cv2.flip(im_v, 0)
    else:
        rec = (im_y.shape[1] - crop_width, 0, crop_width, im_y.shape[0])
        im_y = im_y[:, -crop_width:]
        im_u = im_u[:, -crop_width // 2:]
        im_v = im_v[:, -crop_width // 2:]

    resized_height = MODEL_HEIGHT
    resized_width = MODEL_WIDTH

    source_height = int(0.7 * resized_height)
    extra_height = int((resized_height - source_height) / 2)
    extra_width = int((resized_width - source_height / 2) / 2)
    source_width = int(source_height / 2)

    im_y = im_y[extra_height:-extra_height, extra_width:-extra_width]
    im_u = im_u[extra_height // 2:-extra_height // 2, extra_width // 2:-extra_width // 2]
    im_v = im_v[extra_height // 2:-extra_height // 2, extra_width // 2:-extra_width // 2]
    input_im_y = cv2.resize(im_y, (resized_width, resized_height))
    input_im_v = cv2.resize(im_v, (resized_width // 2, resized_height // 2))
    input_im_u = cv2.resize(im_u, (resized_width // 2, resized_height // 2))

    inputs = np.zeros((1, 6, MODEL_HEIGHT // 2, MODEL_WIDTH // 2), dtype=np.float32)

    inputs[0, 0, :, :] = input_im_y[0:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2]
    inputs[0, 1, :, :] = input_im_y[1:MODEL_HEIGHT:2, 0:MODEL_WIDTH:2]
    inputs[0, 2, :, :] = input_im_y[0:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2]
    inputs[0, 3, :, :] = input_im_y[1:MODEL_HEIGHT:2, 1:MODEL_WIDTH:2]
    inputs[0, 4, :, :] = input_im_u
    inputs[0, 5, :, :] = input_im_v

    inputs = (inputs - 128.) * 0.0078125

    return inputs, rec


def test_images(is_RHD):
    # load dataset
    pandora_file = 'dataset/pandora/pandora.bin'
    pandora = torch.load(pandora_file)
    rlts = {
        'pitch_rlts': [],
        'yaw_rlts': [],
        'roll_rlts': [],
        'distraction_rlts': [],
        'attacked_rlts': []
            }
    for x, pose, position, distraction in tqdm(pandora):
        x = cv2.warpPerspective(x, trans_webcam_to_eon_front_pandora, (WEBCAM_WIDTH, WEBCAM_HEIGHT),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inputs, rec = im_preprocessing(x, is_RHD)

        cv2.rectangle(x, (int(rec[0] + 320 * .015), int(rec[1] + 640 * 0.15)),
                      (int(rec[0] + rec[2] - 320 * 0.15), int(rec[1] + rec[3] - 640 * 0.15)),
                      (0, 255, 0), 5)
        # plt.axis('off')
        # plt.imshow(x)
        # plt.show()

        ort_session = ort.InferenceSession("dmonitoring_model_version_2.onnx",
                                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        outputs = ort_session.run(None, {"input_img": inputs.astype(np.float32)}, )

        _, driverStats = distracted_detect(outputs)
        rlts['pitch_rlts'].append(abs(driverStats.pose.pitch*180/np.pi-pose[2]))
        rlts['yaw_rlts'].append(abs(driverStats.pose.yaw * 180 / np.pi - pose[0]))
        rlts['roll_rlts'].append(abs(driverStats.pose.roll * 180 / np.pi - pose[1]))
        rlts['distraction_rlts'].append(driverStats.distracted == bool(distraction))
        # print('calibrated:', driverStats.pose.pitch*180/np.pi, driverStats.pose.yaw*180/np.pi)

        attack_img = make_attack_img(inputs)
        outputs_attack = ort_session.run(None, {"input_img": attack_img.astype(np.float32)}, )
        _, driverStats_attack = distracted_detect(outputs_attack)
        rlts['attacked_rlts'].append(driverStats_attack.distracted == bool(distraction))
        print(driverStats.distracted, driverStats_attack.distracted, distraction)

    torch.save(rlts, 'test_rlts_on_pandora.bin')
    print('pitch_err, yaw_err, roll_err', np.mean())


if __name__ == "__main__":
    test_images(is_RHD=False)



