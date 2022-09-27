from dataset_pandora import PandoraData
from image_processing import im_preprocessing, make_frame
import onnxruntime as ort
import cv2
import torch
import numpy as np
from demonitoring import distracted_detect
from attack import make_attack_img
from camera_param import trans_webcam_to_eon_front_pandora, WEBCAM_WIDTH, WEBCAM_HEIGHT
from tqdm import tqdm
import argparse
import time
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("TkAgg")
import random
import matplotlib.pyplot as plt


def test_images(args):
    # load dataset
    pandora_file = 'dataset/pandora/pandora.bin'
    pandora = torch.load(pandora_file)
    # pandora_loader = DataLoader(pandora, batch_size=1, pin_memory=True, shuffle=True)
    print(f'Total number of data Pandora : {len(pandora)}')

    # define result structure
    rlts = {
        'pitch_rlts': [],
        'yaw_rlts': [],
        'roll_rlts': [],
        'distraction_rlts': [],
        'attacked_rlts': []
            }

    # load model
    ort_session = ort.InferenceSession("models/dmonitoring_model_version_2.onnx",
                                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    num_distract = 0
    num_attack = 0
    time_cost = 0
    mse_total = 0
    max_eps = 0.0
    for x, pose, position, distraction in tqdm(pandora):
    # x, pose, position, distraction = pandora[1050]
        num_distract += int(np.sum(distraction))
        im_ori = x
        x = cv2.warpPerspective(x, trans_webcam_to_eon_front_pandora, (WEBCAM_WIDTH, WEBCAM_HEIGHT),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inputs, rec = im_preprocessing(x, args.Is_RHD)
        # crop = np.copy(x[int(rec[1] + 640 * 0.15):int(rec[1] + rec[3] - 640 * 0.15),
        #                int(rec[0] + 320 * .15):int(rec[0] + rec[2] - 320 * 0.15), :])

        cv2.rectangle(x, (int(rec[0] + 320 * .15), int(rec[1] + 640 * 0.15)),
                      (int(rec[0] + rec[2] - 320 * 0.15), int(rec[1] + rec[3] - 640 * 0.15)),
                      (0, 255, 0), 5)
        # plt.imshow(im_ori)
        # plt.show()
        outputs = ort_session.run(None, {"input_img": inputs.astype(np.float32)}, )

        _, driverStats = distracted_detect(outputs)
        rlts['pitch_rlts'].append(abs(driverStats.pose.pitch*180/np.pi+pose[2]))
        rlts['yaw_rlts'].append(abs(driverStats.pose.yaw * 180 / np.pi - pose[0]))
        rlts['roll_rlts'].append(abs(driverStats.pose.roll * 180 / np.pi + pose[1]))
        rlts['distraction_rlts'].append(driverStats.distracted == bool(distraction))
        # print('pitch raw, pred,  calibrated:', pose[2],  driverStats.pose.pitch*180/np.pi)
        # print('yaw raw, pred,  calibrated:', pose[0], driverStats.pose.yaw * 180 / np.pi)
        # print('roll raw, pred,  calibrated:', pose[1], driverStats.pose.roll * 180 / np.pi)

        if driverStats.distracted and bool(distraction):
            start = time.time()
            attack_img = make_attack_img(inputs, attack_method=args.attack_method, eps=10/255, c=args.c)
            end = time.time()
            max_eps = np.maximum(np.max(np.abs(attack_img-inputs)), max_eps)
            time_cost += end-start
            outputs_attack = ort_session.run(None, {"input_img": attack_img.astype(np.float32)}, )
            crop, attack_img = make_frame(attack_img, inputs, rec)
            #
            # fig, axes = plt.subplots(1, 3, figsize=(16, 8))
            # axes[0].imshow(crop)
            # axes[0].axis('off')
            # axes[0].set_title('Distracted', fontsize=20)
            # axes[2].imshow(attack_img)
            # axes[2].axis('off')
            # axes[2].set_title('Non distracted', fontsize=20)
            # axes[1].imshow(attack_img-crop)
            # axes[1].axis('off')
            # fig.tight_layout(pad=0.5)
            # plt.savefig('img_rlts/attack_sample1000')
            perturbation = attack_img-crop
            perturbation[perturbation > 200] = 255-perturbation[perturbation > 200]
            mse = np.linalg.norm(perturbation)**2/crop.size
            mse_total += mse
            num_attack += 1
            _, driverStats_attack = distracted_detect(outputs_attack)
            rlts['attacked_rlts'].append(driverStats_attack.distracted == bool(distraction))
            # print(driverStats.distracted, driverStats_attack.distracted, distraction)

    print('pitch_err:', np.mean(np.array(rlts['pitch_rlts'])), np.std(np.array(rlts['pitch_rlts'])))
    print('yaw_err:', np.mean(np.array(rlts['yaw_rlts'])), np.std(np.array(rlts['yaw_rlts'])))
    print('roll_err:', np.mean(np.array(rlts['roll_rlts'])), np.std(np.array(rlts['roll_rlts'])))
    print('distraction_acc_total:', np.mean(np.array(rlts['distraction_rlts'])))
    print('number of distracted samples', num_distract)
    print('number of attacked distracted samples', num_attack)
    print('time cost', time_cost/num_attack)
    print('mean mse', mse_total/num_attack)
    print('max eps', max_eps)
    print('distraction decay when applying attack (only on well distinguished samples)',
          np.mean(np.array(rlts['attacked_rlts'])))

    torch.save(rlts, f'results/test_rlts_on_pandora_{args.attack_method}_rectified_robustness.bin')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('dm model attack')
    parser.add_argument(
        '--Is-RHD',
        default=False,
        type=bool,
        help='Choose parameter is-rhd, Ture is for right hand drive, False is for left hand drive'
    )
    parser.add_argument(
        '--attack-method',
        default='FGSM',
        help='choose attack model used for generating adversarial sample, the default is FGSM, methods choices: '
             '{FGSM, CW, PGD, PGDL2, APGD, AutoAttack, DeepFool}'
    )
    parser.add_argument(
        '--perturbation', '-p',
        default=10/255,
        type=float,
        help='Norm_2 or norm_inf of perturbation, default is 10/255 for norm_inf'
    )
    parser.add_argument(
        '--coef', '-c',
        default=1,
        type=float,
        help='coefficient in CW methods, default is 1'
    )
    args = parser.parse_args()

    # 'FGSM', 'CW', 'PGD', 'APGD', 'DeepFool', 'AutoAttack',
    methods = ['CW']
    eps_set = [1/255, 2/255, 3/255, 4/255, 5/255, 8/255, 10/255]
    c_set = np.logspace(-2, 3, 10)
    seed = 2
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for m in methods:
        args.attack_method = m
        print(args.attack_method)
        for c in c_set:
            args.c = c
            print(args.c)
            test_images(args)



