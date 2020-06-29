
import cv2
from torch import long ,tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import PIL
class MaskDataset(Dataset):

    def __init__(self,data_frame):
        self.data_frame=data_frame
        self.transformations=Compose([ToPILImage(),
                                      Resize((100,100)),
                                      ToTensor()
                                      ])
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        row=self.data_frame.iloc[key]

        return {'image': self.transformations(cv2.imread(row['image'])),
                'target': tensor([row['target']],dtype=long)
                }
    def __len__(self):
        return len(self.data_frame.index)





