import cv2
import argparse
from head_pose_detect import head_pose_detect
from demonitoring import distracted_detect
from camera_param import trans_webcam_to_eon_front, WEBCAM_WIDTH, WEBCAM_HEIGHT

import torch
import numpy as np

from playsound import playsound
import threading

global alert
alert = False


def play(args):
    stream = cv2.VideoCapture(0)
    stream.set(3, 1280)
    stream.set(4, 720)

    if stream.isOpened is False:
        print('unable to read data from camera')

    # output = cv2.VideoWriter('processed_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
    #                          (frame_width, frame_height))

    while (True):

        ret, frame = stream.read()
        if ret:
            frame = cv2.warpPerspective(frame, trans_webcam_to_eon_front, (WEBCAM_WIDTH, WEBCAM_HEIGHT),
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            outputs, attack_outputs, frame_attack, rec = head_pose_detect(frame, args.Is_RHD)
            dmonitoringResults, driverStats = distracted_detect(outputs)
            dmonitoringResults_attack, driverStats_attack = distracted_detect(attack_outputs)
            #
            head_position_x = int((dmonitoringResults.face_position[0]+0.5)*(372-320*0.3)-372+320*0.15+1152)
            head_position_y = int((dmonitoringResults.face_position[1]+0.5)*(864-640*0.3)+640*0.15)
            cv2.rectangle(frame, (head_position_x-40, head_position_y-100), (head_position_x+50, head_position_y+10), (255, 0, 0), 5)

            cv2.putText(frame, f'Face_prob: {dmonitoringResults.face_prob}', (30, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Partial_face: {dmonitoringResults.partial_face}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Face_orientation: {dmonitoringResults.face_orientation*180/np.pi}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f'Face_orientationStd: {dmonitoringResults.face_orientation_meta*180/np.pi}', (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f'Face_position: {dmonitoringResults.face_position}', (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(frame, f'Sunglasses: {dmonitoringResults.sunglasses_prob}', (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Eye_prob: {dmonitoringResults.left_eye_prob, dmonitoringResults.right_eye_prob}',
                        (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Blink_prob: {dmonitoringResults.left_blink_prob, dmonitoringResults.right_blink_prob}',
                        (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Is_distracted: '
                               f'{driverStats.distracted}',(30, 320),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f'Is_distracted_after_attack: '
                               f'{driverStats_attack.distracted}', (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if driverStats.distracted:
                cv2.putText(frame, f'Alert: KEEP EYES ON ROAD ', (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.rectangle(frame, (int(rec[0]+320*.015), int(rec[1]+640*0.15)),
                          (int(rec[0]+rec[2]-320*0.15), int(rec[1]+rec[3]-640*0.15)),
                          (0, 255, 0), 5)
            # cv2.rectangle(frame1, (rec[0], rec[1]), (rec[0] + rec[2], rec[1] + rec[3]), (0, 255, 0), 5)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)
            # output.write(frame)
            # frame = cv2.resize(frame, (frame_width*2, frame_height*2))
            show_img = np.hstack((frame, frame_attack))
            cv2.imshow('Distraction detection', show_img)
            if cv2.waitKey(1) and 0xFF == 'q':
                break
        else:
            break

    stream.release()
    # output.release()
    cv2.destroyAllWindows()


def alarm():
    if alert:
        playsound('mixkit-classic-alarm-995.wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Driver Monitoring')
    parser.add_argument(
        '--Is-RHD',
        default=False,
        type=bool,
        help='Choose parameter is-rhd, Ture is for right hand drive, False is for left hand drive'
    )
    args = parser.parse_args()
    # play()
    p1 = threading.Thread(target=play, args=(args,))
    p1.start()
    #
    p2 = threading.Thread(target=alarm, args=())
    p2.start()
    #
    p1.join()
    p2.join()
