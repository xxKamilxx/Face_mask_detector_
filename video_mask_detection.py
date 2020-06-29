from pathlib import Path
import cv2
import torch
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from train_mask_detector import MaskDetector
from face_detector import FaceDetector
# cascade_classifier = cv2.CascadeClassifier('C:/Users/kamil/Desktop/MLnauka/haar/haarcascades/haarcascade_frontalface_default.xml')
model=MaskDetector()
model.load_state_dict(torch.load('C:/Users/kamil/Desktop/mask_detector_pytorch/_ckpt_epoch_5.ckpt')['state_dict'],strict=False)
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model=model.to(device)
model.eval()
font=cv2.FONT_HERSHEY_SIMPLEX
transformations = Compose([
    ToPILImage(),
    Resize((100, 100)),
    ToTensor(),
])

faceDetector = FaceDetector(
        prototype='C:/Users/kamil/Desktop/mask_detector_pytorch/deploy.prototxt.txt',
        model='C:/Users/kamil/Desktop/mask_detector_pytorch/res10_300x300_ssd_iter_140000.caffemodel',
    )



labels=['No mask','Mask']
labels_color=[(255,0,0),(0,255,0)]
camera=cv2.VideoCapture(1)

while camera.isOpened():
    _,cam_frame=camera.read()

    # faces = cascade_classifier.detectMultiScale(cam_frame, minNeighbors=1)
    faces = faceDetector.detect(cam_frame)

    for face in faces:
        x, y, w, h=face
        x, y = max(x, 0), max(y, 0)
        detected_face = cam_frame[int(y):int(y + h), int(x):int(x + w)]
        output=model(transformations(detected_face).unsqueeze(0))
        _,predicted=torch.max(output.data,1)
        print(labels[int(predicted[0])])



        cv2.putText(cam_frame,labels[int(predicted[0])],(x,y), font,2, labels_color[int(predicted[0])], 3)


    cv2.imshow("mask detector",cam_frame)
    key=cv2.waitKey(30)
    if key==27:
        break
camera.release()
cv2.destroyAllWindows()




