import numpy as np
import torch
import torch.nn as nn


RESIZED_FOCAL = 320.0
H, W, FULL_W = 320, 160, 426


class DMonitoringResult:
    def __init__(self, output):
        self.face_orientation = output[0, 0:3]
        self.face_orientation_meta = nn.Softplus()(torch.from_numpy(output[0, 6:9])).numpy()
        self.face_position = output[0, 3:5]
        self.face_position_meta = nn.Softplus()(torch.from_numpy(output[0, 9:11])).numpy()
        self.face_prob = output[0, 12]
        self.left_eye_prob = output[0, 21]
        self.right_eye_prob = output[0, 30]
        self.left_blink_prob = output[0, 31]
        self.right_blink_prob = output[0, 32]
        self.sunglasses_prob = output[0, 33]
        self.poor_vision = output[0, 34]
        self.partial_face = output[0, 35]
        self.distracted_pose = output[0, 36]
        self.distracted_eyes = output[0, 37]
        # self.occluded_prob = output[38]


class DRIVER_MONITOR_SETTINGS():

    def __init__(self):
        self._AWARENESS_TIME = 35.  # passive wheeltouch total timeout
        self._AWARENESS_PRE_TIME_TILL_TERMINAL = 12.
        self._AWARENESS_PROMPT_TIME_TILL_TERMINAL = 6.
        self._DISTRACTED_TIME = 11.  # active monitoring total timeout
        self._DISTRACTED_PRE_TIME_TILL_TERMINAL = 8.
        self._DISTRACTED_PROMPT_TIME_TILL_TERMINAL = 6.

        self._FACE_THRESHOLD = 0.5
        self._PARTIAL_FACE_THRESHOLD = 0.765  # if TICI else 0.43
        self._EYE_THRESHOLD = 0.61  # if TICI else 0.55
        self._SG_THRESHOLD = 0.89   # if TICI else 0.86
        self._BLINK_THRESHOLD = 0.82   # if TICI else 0.588
        self._BLINK_THRESHOLD_SLACK = 0.9  # if TICI else 0.77
        self._BLINK_THRESHOLD_STRICT = self._BLINK_THRESHOLD

        self._POSE_PITCH_THRESHOLD = 0.3237
        self._POSE_PITCH_THRESHOLD_SLACK = 0.3657
        self._POSE_PITCH_THRESHOLD_STRICT = self._POSE_PITCH_THRESHOLD
        self._POSE_YAW_THRESHOLD = 0.3109
        self._POSE_YAW_THRESHOLD_SLACK = 0.4294
        self._POSE_YAW_THRESHOLD_STRICT = self._POSE_YAW_THRESHOLD
        self._PITCH_NATURAL_OFFSET = 0.057 # initial value before offset is learned
        self._YAW_NATURAL_OFFSET = 0.11 # initial value before offset is learned
        self._PITCH_MAX_OFFSET = 0.124
        self._PITCH_MIN_OFFSET = -0.0881
        self._YAW_MAX_OFFSET = 0.289
        self._YAW_MIN_OFFSET = -0.0246

        self._POSESTD_THRESHOLD = 0.38   # if TICI else 0.3
        # self._HI_STD_FALLBACK_TIME = int(10  / self._DT_DMON)  # fall back to wheel touch if model is uncertain for 10s
        self._DISTRACTED_FILTER_TS = 0.25  # 0.6Hz

        self._POSE_CALIB_MIN_SPEED = 13  # 30 mph
        # self._POSE_OFFSET_MIN_COUNT = int(60 / self._DT_DMON)  # valid data counts before calibration completes, 1min cumulative
        # self._POSE_OFFSET_MAX_COUNT = int(360 / self._DT_DMON)  # stop deweighting new data after 6 min, aka "short term memory"

        self._RECOVERY_FACTOR_MAX = 5.  # relative to minus step change
        self._RECOVERY_FACTOR_MIN = 1.25  # relative to minus step change

        self._MAX_TERMINAL_ALERTS = 3  # not allowed to engage after 3 terminal alerts
        # self._MAX_TERMINAL_DURATION = int(30 / self._DT_DMON)  # not allowed to engage after 30s of terminal alerts


class DriverPose():

    def __init__(self):
        self.yaw = 0.
        self.pitch = 0.
        self.roll = 0.
        self.yaw_std = 0.
        self.pitch_std = 0.
        self.roll_std = 0.
        self.pitch_offseter = 0
        self.yaw_offseter = 0
        self.low_std = True
        self.cfactor_pitch = 1.
        self.cfactor_yaw = 1.


class DriverBlink():
  def __init__(self):
    self.left_blink = 0.
    self.right_blink = 0.
    self.cfactor = 1.


def face_orientation_from_net(angles_desc, pos_desc):
    # the output of these angles are in device frame
    # so from driver's perspective, pitch is up and yaw is right

    pitch_net, yaw_net, roll_net = angles_desc

    face_pixel_position = ((pos_desc[0] + .5)*W - W + FULL_W, (pos_desc[1]+.5)*H)
    yaw_focal_angle = 0  # np.arctan2(face_pixel_position[0] - FULL_W//2, RESIZED_FOCAL)
    pitch_focal_angle = 0  # np.arctan2(face_pixel_position[1] - H//2, RESIZED_FOCAL)

    pitch = pitch_net + pitch_focal_angle
    yaw = -yaw_net + yaw_focal_angle

    # # no calib for roll
    # pitch -= rpy_calib[1]
    # yaw -= rpy_calib[2] * (1 - 2 * int(is_rhd))  # lhd -> -=, rhd -> +=
    return roll_net, pitch, yaw


class DistractedType:
    NOT_DISTRACTED = 0
    BAD_POSE = 1
    BAD_BLINK = 2


class DriverStatus():
    def __init__(self, rhd=False, settings=DRIVER_MONITOR_SETTINGS()):
        # init policy settings
        self.settings = settings

        # init driver status
        self.is_rhd_region = rhd
        self.pose = DriverPose()
        self.pose_calibrated = False
        self.blink = DriverBlink()
        self.awareness = 1.
        self.awareness_active = 1.
        self.awareness_passive = 1.
        self.driver_distracted = False
        self.face_detected = False
        self.face_partial = False
        self.terminal_alert_cnt = 0
        self.terminal_time = 0
        self.step_change = 0.
        self.active_monitoring_mode = True
        self.is_model_uncertain = False
        self.hi_stds = 0
        self.threshold_pre = self.settings._DISTRACTED_PRE_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME
        self.threshold_prompt = self.settings._DISTRACTED_PROMPT_TIME_TILL_TERMINAL / self.settings._DISTRACTED_TIME

    def _is_driver_distracted(self, pose, blink):
        if not self.pose_calibrated:
            pitch_error = pose.pitch - self.settings._PITCH_NATURAL_OFFSET
            yaw_error = pose.yaw - self.settings._YAW_NATURAL_OFFSET
        else:
            pitch_error = pose.pitch - min(max(self.pose.pitch_offseter.filtered_stat.mean(),
                                               self.settings._PITCH_MIN_OFFSET), self.settings._PITCH_MAX_OFFSET)
            yaw_error = pose.yaw - min(max(self.pose.yaw_offseter.filtered_stat.mean(),
                                           self.settings._YAW_MIN_OFFSET), self.settings._YAW_MAX_OFFSET)

        pitch_error = 0 if pitch_error > 0 else abs(pitch_error)  # no positive pitch limit
        yaw_error = abs(yaw_error)

        if pitch_error > self.settings._POSE_PITCH_THRESHOLD * pose.cfactor_pitch or \
                yaw_error > self.settings._POSE_YAW_THRESHOLD * pose.cfactor_yaw:
            return DistractedType.BAD_POSE
        elif (blink.left_blink + blink.right_blink) * 0.5 > self.settings._BLINK_THRESHOLD * blink.cfactor:
            return DistractedType.BAD_BLINK
        else:
            return DistractedType.NOT_DISTRACTED

    def get_pose(self, driver_state):
        if not all(len(x) > 0 for x in (driver_state.face_orientation, driver_state.face_position,
                                        driver_state.face_orientation_meta, driver_state.face_position_meta)):
            return

        self.face_partial = driver_state.partial_face > self.settings._PARTIAL_FACE_THRESHOLD
        self.face_detected = driver_state.face_prob > self.settings._FACE_THRESHOLD or self.face_partial
        self.pose.roll, self.pose.pitch, self.pose.yaw = face_orientation_from_net(driver_state.face_orientation,
                                                                                   driver_state.face_position)
        self.pose.pitch_std = driver_state.face_orientation_meta[0]
        self.pose.yaw_std = driver_state.face_orientation_meta[1]
        # self.pose.roll_std = driver_state.faceOrientationStd[2]
        model_std_max = max(self.pose.pitch_std, self.pose.yaw_std)
        self.pose.low_std = model_std_max < self.settings._POSESTD_THRESHOLD and not self.face_partial
        self.blink.left_blink = driver_state.left_blink_prob * (
                    driver_state.left_eye_prob > self.settings._EYE_THRESHOLD) * (
                                            driver_state.sunglasses_prob < self.settings._SG_THRESHOLD)
        self.blink.right_blink = driver_state.right_blink_prob * (
                    driver_state.right_eye_prob > self.settings._EYE_THRESHOLD) * (
                                             driver_state.sunglasses_prob < self.settings._SG_THRESHOLD)

        self.driver_distracted = self._is_driver_distracted(self.pose, self.blink) > 0 and \
                                 driver_state.face_prob > self.settings._FACE_THRESHOLD and self.pose.low_std
        self.distracted = self.driver_distracted or not self.pose.low_std or self.face_partial

        # update offseter
        # only update when driver is actively driving the car above a certain speed
        # self.pose_calibrated = self.pose.pitch_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT and \
        #                        self.pose.yaw_offseter.filtered_stat.n > self.settings._POSE_OFFSET_MIN_COUNT

        # self.is_model_uncertain = self.hi_stds > self.settings._HI_STD_FALLBACK_TIME
        # if self.face_detected and not self.pose.low_std and not self.driver_distracted:
        #     self.hi_stds += 1
        # elif self.face_detected and self.pose.low_std:
        #     self.hi_stds = 0


