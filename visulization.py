import cv2
from PIL import Image
from torchvision import transforms
import numpy as np


def read_annotation(file):

    angles = {}
    head_pos = {}
    with open(file=file, mode="r") as f:
        lines = f.readlines()
        for l in lines:
            data = l.rstrip().split('\t')
            angles[data[1]] = np.array([float(data[2]), float(data[3]), float(data[4])], dtype=np.float32)  # [yaw, roll, pitch]
            head_pos[data[1]] = np.array([float(data[107])/1920, float(data[108])/1080.0], dtype=np.float32)  # [x, y]

    return angles, head_pos


root = '/home/flavie2/Desktop/driver_monitor/driver_monitor/dataset/pandora/08/RGB/000050_RGB.png'
annotation = 'dataset/pandora/08/data.txt'
angles, head_pos = read_annotation(annotation)

img = cv2.imread(root)
nx, ny, c = img.shape
img = cv2.resize(img, (1320, 1320))
ang = angles['000050']
pos = head_pos['000050']
im = cv2.rectangle(img, (int(pos[0]*1320-200), int(pos[1]*1320-200)), (int(pos[0]*1320+200), int(pos[1]*1320+200)), color=(255, 0, 0), thickness=5)
cv2.imshow('images', im)
cv2.waitKey(0)
cv2.destroyAllWindows()






