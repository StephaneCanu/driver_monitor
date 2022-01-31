import threading

import cv2
import os
import numpy as np
from torchvision import transforms
from model import HeadLocModel, Poseidon, load_weight
import torch
from PIL import Image
from playsound import playsound
from deepface import DeepFace
import onnxruntime as ort
from demonitoring import DriverStatus, DMonitoringResult
import threading

global alert
alert = False


def play():
    stream = cv2.VideoCapture(0)

    if stream.isOpened is False:
        print('unable to read data from camera')

    frame_width = int(stream.get(3))
    frame_height = int(stream.get(4))

    # output = cv2.VideoWriter('processed_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
    #                          (frame_width, frame_height))

    while (True):

        ret, frame = stream.read()
        if ret:
            # transform = transforms.Compose([
            #     transforms.Resize((64, 64)),
            #     transforms.ToTensor()
            # ])
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
            # faces = face_cascade.detectMultiScale(image=frame, scaleFactor=1.1, minNeighbors=4)

            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

            # im_y, im_x, ch = frame.shape
            #
            # # # head detector with deep model used in poseidon architecture
            # # im = cv2.cvtColor(frame, cv2.COLOR_BGR2GREY)
            # # head_loc_model = load_weight(HeadLocModel(channel=1).to(device)).eval()
            # # head_loc = head_loc_model(transform(Image.fromarray(im)).unsqueeze(0).to('cuda'))
            # # x, y = int(head_loc[0, 0] * im_x), int(head_loc[0, 1] * im_y)
            # # cv2.rectangle(frame, (int(im_x / 2) + x - 150, im_y + y - 150), (int(im_x / 2) + x + 150, im_y + y + 150),
            # #               (255, 0, 0), 4)
            # backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
            # _, region = DeepFace.detectFace(im, target_size=(224, 224), detector_backend=backends[1])
            # if len(region)>0:
            #     x, y, w, h = region
            #     point1 = (max(x-50, 0), max(y-50, 0))
            #     point2 = (min(x+w+50, frame_width), min(y+h+50, frame_height))
            #     cv2.rectangle(frame, point1, point2, (255, 255), 4)
            #
            #     head = frame[max(y-50, 0):min(y+h+50, frame_height), max(x-50, 0):min(x+w+50, frame_width)]
            # else:
            #     head = frame
            head = cv2.cvtColor(cv2.resize(im, (320, 640)), cv2.COLOR_RGB2YUV)
            head_y, head_u, head_v = cv2.split(head)
            head_u = cv2.resize(head_u, (head_u.shape[1]//2, head_u.shape[0]//2))
            head_v = cv2.resize(head_v, (head_v.shape[1]//2, head_v.shape[0]//2))
            yr, yc = head_y.shape
            input = np.zeros((1, 6, int(yr/2), int(yc/2)))
            input[0, 0, :, :] = head_y[0:yr:2, 0:yc:2]
            input[0, 1, :, :] = head_y[1:yr:2, 0:yc:2]
            input[0, 2, :, :] = head_y[0:yr:2, 1:yc:2]
            input[0, 3, :, :] = head_y[1:yr:2, 1:yc:2]
            input[0, 4, :, :] = head_u   # head[:int(yr/4), :, 1].reshape(int(yr/2), int(yc/2))
            input[0, 5, :, :] = head_v   # head[:int(yr/4), :, 2].reshape(int(yr/2), int(yc/2))

            input = (input - 128.) * 0.0078125

            ort_session = ort.InferenceSession("dmonitoring_model.onnx",
                                               providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            outputs = ort_session.run(None, {"input_img": input.astype(np.float32)},)
            dmonitoringResults = DMonitoringResult(outputs[0])
            driverStats = DriverStatus()
            driverStats.get_pose(dmonitoringResults)
            #
            # head = Image.fromarray(head)
            # head = transform(head)
            # head_pos_model = load_weight(Poseidon(channel=1)).to(device).eval()
            # head_pos = head_pos_model(head.unsqueeze(1).to(device))
            # cv2.putText(frame, f'yaw: {head_pos[0, 0].item()*180/np.pi}', (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # cv2.putText(frame, f'roll: {head_pos[0, 1].item()*180/np.pi}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # cv2.putText(frame, f'pitch: {head_pos[0, 2].item()*180/np.pi}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # if head_pos[0, 0] > np.pi/4 or head_pos[0, 1] > np.pi/4 or head_pos[0, 2] > np.pi/4:
            #     cv2.putText(frame, f'Distraction', (30, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255),
            #                 2, cv2.LINE_AA)

            cv2.putText(frame, f'Face_prob: {dmonitoringResults.face_prob}', (30, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Partial_face: {dmonitoringResults.partial_face}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Face_orientation: {dmonitoringResults.face_orientation}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Face_orientationStd: {dmonitoringResults.face_orientation_meta}', (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Face_position: {dmonitoringResults.face_position}', (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, f'Sunglasses: {dmonitoringResults.sunglasses_prob}', (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Eye_prob: {dmonitoringResults.left_eye_prob, dmonitoringResults.right_eye_prob}',
                        (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Blink_prob: {dmonitoringResults.left_blink_prob, dmonitoringResults.right_blink_prob}',
                        (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Is_distracted: '
                               f'{driverStats.distracted}',(30, 320),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if driverStats.distracted:
                cv2.putText(frame, f'Alert: KEEP EYES ON ROAD ', (30, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # if head_pos[0, 0] > np.pi/4 or head_pos[0, 1] > np.pi/4 or head_pos[0, 2] > np.pi/4:
            #     cv2.putText(frame, f'Distraction', (30, 140), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255),
            #                 2, cv2.LINE_AA)

            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # output.write(frame)
            frame = cv2.resize(frame, (frame_width*2, frame_height*2))
            cv2.imshow('Distraction detection', frame)
            # cv2.waitKey(0)
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
    # play()
    p1 = threading.Thread(target=play, args=())
    p1.start()
    #
    p2 = threading.Thread(target=alarm, args=())
    p2.start()
    #
    p1.join()
    p2.join()
