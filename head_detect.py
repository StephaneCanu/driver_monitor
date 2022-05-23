
from deepface import DeepFace
import cv2
from torchvision import transforms
from poseidon_model import HeadLocModel, Poseidon, load_weight
from PIL import Image


def head_detect_deep(img=None, backend='mtcnn'):
    # backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    _, region = DeepFace.detectFace(img, target_size=(224, 224), detector_backend=backend)
    if len(region) > 0:
        x, y, w, h = region
    else:
        print('no head detected')
        x, y = None, None

    return x, y, w, h


def head_detect_opencv(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
    else:
        x, y = None, None
        print('no head detected')
    return x, y


def head_detect_Posidon(img, device='cpu'):
    # head detector with deep model used in poseidon architecture
    transform = transforms.Compose([
            transforms.Resize((132, 132)),
            transforms.ToTensor(),
        ])
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GREY)
    im_y, im_x, ch = im.shape
    head_loc_model = load_weight(HeadLocModel(channel=1).to(device)).eval()
    head_loc = head_loc_model(transform(Image.fromarray(im)).unsqueeze(0).to(device))
    x, y = int(head_loc[0, 0] * im_x), int(head_loc[0, 1] * im_y)
    return x, y


def head_detect(im, method, device='cpu'):

    if method == 'deep':
        return head_detect_deep(im, backend='ssd')
    elif method == 'opencv':
        return head_detect_opencv(im)
    elif method == 'Poseidon':
        return head_detect_Posidon(im, device=device)