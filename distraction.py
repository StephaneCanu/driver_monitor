import numpy as np
import torch
import torch.nn as nn


PARTIAL_FACE_THRESHOLD = 0.43
FACE_THRESHOLD = 0.5
POSESTD_THRESHOLD = 0.3
SG_THRESHOLD = 0.86
EYE_THRESHOLD = 0.55
PITCH_NATURAL_OFFSET = 0.057
YAW_NATURAL_OFFSET = 0.11
POSE_PITCH_THRESHOLD = 0.3237
POSE_YAW_THRESHOLD = 0.3109
BLINK_THRESHOLD = 0.588

W, H, BW, BH, FULL_W = 372, 864, 320*0.15, 640*0.15, 1152


class Is_Distraction(nn.Module):

    def __init__(self, device):
        super(Is_Distraction, self).__init__()
        self.f_x = torch.Tensor([1403]).to(device)
        self.f_y = torch.Tensor([1266]).to(device)

    def forward(self, y):
        y = y.squeeze(1)
        self.face_partial = y[:, 35:36] - PARTIAL_FACE_THRESHOLD
        self.face = y[:, 12:13] - FACE_THRESHOLD
        self.face_detected = nn.ReLU()(self.face_partial) + nn.ReLU()(self.face)
        self.not_face_detected = nn.ReLU()(-self.face_partial)*nn.ReLU()(-self.face)

        self.pose_pitch_std = nn.ReLU()(POSESTD_THRESHOLD - y[:, 6:7])
        self.pose_yaw_std = nn.ReLU()(POSESTD_THRESHOLD - y[:, 7:8])
        self.low_std = self.pose_pitch_std * self.pose_yaw_std * nn.ReLU()(-self.face_partial)

        self.sunglass = (SG_THRESHOLD - y[:, 32:33]) / (SG_THRESHOLD - y[:, 32:33])
        self.right_eye = (y[:, 30:31] - EYE_THRESHOLD) / (y[:, 30:31] - EYE_THRESHOLD)
        self.left_eye = (y[:, 21:22] - EYE_THRESHOLD) / (y[:, 21:22] - EYE_THRESHOLD)
        self.right_blink = y[:, 32:33] * self.right_eye * self.sunglass
        self.left_blink = y[:, 31:32] * self.left_eye * self.sunglass

        self.face_position_x = (y[:, 3:4] + 0.5) * (W - 2 * BW) + FULL_W - W + BW
        self.face_position_y = (y[:, 4:5] + 0.5) * (H - 2 * BH) + BH

        self.pose_pitch = y[:, 0:1]  # + torch.atan2(self.face_position_y - H // 2, self.f_y)
        self.pose_yaw = -y[:, 1:2]  #+ torch.atan2(self.face_position_x - FULL_W // 2, self.f_x)
        self.pose_pitch_err = nn.ReLU()(-self.pose_pitch + PITCH_NATURAL_OFFSET)
        self.pose_yaw_err = abs(self.pose_yaw - YAW_NATURAL_OFFSET)

        self.bad_pose = nn.ReLU()(self.pose_pitch_err-POSE_PITCH_THRESHOLD)+nn.ReLU()(self.pose_yaw_err-POSE_YAW_THRESHOLD)
        self.good_pose = nn.ReLU()(-self.pose_pitch_err+POSE_PITCH_THRESHOLD)*nn.ReLU()(-self.pose_yaw_err+POSE_YAW_THRESHOLD)
        self.bad_blink = nn.ReLU()((self.left_blink + self.right_blink) * 0.5 - BLINK_THRESHOLD)
        self.good_blink = nn.ReLU()(-(self.left_blink + self.right_blink) * 0.5 + BLINK_THRESHOLD)

        self.distracted = (self.bad_pose+self.bad_blink)*self.low_std*nn.ReLU()(self.face) + self.not_face_detected
        # self.non_distracted = self.face_detected*(self.good_pose*self.good_blink)

        # return self.distracted
        return torch.hstack((nn.ReLU()(self.face)+nn.ReLU()(-self.face_partial),
                             nn.ReLU()(-self.face)*nn.ReLU()(self.face_partial)))

