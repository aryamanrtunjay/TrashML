#from gpiozero import AngularServo
from time import sleep
import cv2
import torch
import numpy as np

cam = cv2.VideoCapture(0)

#servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained = True)
model.head = torch.nn.Linear(in_features = 768, out_features = 2, bias = True)
model.head_dist = torch.nn.Linear(in_features = 768, out_features = 2, bias = True)
model.load_state_dict(torch.load('./model.pickle', map_location=torch.device('cpu')))
model.eval()

while(True):
    ret, image = cam.read()
    print(ret)
    if ret:
        cv2.imshow('test', image)
        img = cv2.resize(image, (248, 248))
        print(img.shape)
        image_np = np.array(img)
        t = np.transpose(image_np, (2, 0, 1))
        tensor = torch.from_numpy(t)
        print(model(t))

    k = cv2.waitKey(1)
    if k != -1:
        break
    sleep(5)
    '''servo.angle = 90
    sleep(2)
    servo.angle = 0
    sleep(2)
    servo.angle = 90'''
cam.release()
cv2.destroyAllWindows()