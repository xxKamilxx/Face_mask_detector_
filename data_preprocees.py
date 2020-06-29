
import os
import cv2
from pathlib import Path
import pandas as pd
import pickle

no_mask_dataset=Path('C:/Users/kamil/Desktop/RMFD/self-built-masked-face-recognition-dataset/AFDB_face_dataset')
mask_dataset=Path('C:/Users/kamil/Desktop/RMFD/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset')
new="C:/Users/kamil/Desktop/RMFD/self-built-masked-face-recognition-dataset/masks"
mask_df=pd.DataFrame()
l1,l2=0,0


for root,dir,file in os.walk(new,topdown=False):
    for name in file:
        try:
            img_pth = os.path.join(root, name)
            print(img_pth)
            img = cv2.imread(img_pth)
            cv2.imshow("a", img)
            mask_df = mask_df.append({'image': str(img_pth),
                                            'target':1},ignore_index=True)
            l1+=1
        except:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')




for root,dir,file in os.walk(no_mask_dataset,topdown=False):
    for name in file:
        if l2>=60000:
            break
        else:
            img_pth = os.path.join(root, name)
            print(img_pth)
            img=cv2.imread(img_pth)
            cv2.imshow("a",img)
            mask_df = mask_df.append({'image': str(img_pth),
                                        'target':0},ignore_index=True)
            l2+=1


print(l1,l2)

filename="masks_df.pkl"
pickle.dump(mask_df,open(filename,'wb'))





