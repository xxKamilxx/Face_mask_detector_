from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import pytorch_lightning as pl
import torch
from dataset_pytotch import*
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU,
                      Sequential)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger



class MaskDetector(pl.LightningModule):
    def __init__(self, data_path: Path=None):
        super(MaskDetector, self).__init__()
        self.data_path=data_path

        self.mask_df=None
        self.train_df=None
        self.validation_df=None
        self.crossEntrophyloss=None
        self.learning_rate= 0.00001

        self.convLayer1= convLayer1 = Sequential(
            Conv2d(3,32,kernel_size=(3,3),padding=(1,1)),
            ReLU(),
            MaxPool2d(kernel_size=(2,2))
        )

        self.convLayer2 = convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer3 = convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3,3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.linearLayers= linearLayers = Sequential(
            Linear(in_features=2048,out_features=1024 ),
            ReLU(),
            Linear(in_features=1024,out_features=2)
        )

        # initialize weights
        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            for layer in sequential.children():
                if isinstance(layer,(Linear,Conv2d)):
                    init.xavier_uniform_(layer.weight)

    def forward(self, x:Tensor):
        out=self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out=out.view(-1,2048)
        out = self.linearLayers(out)
        return out


    def prepare_data(self) -> None:
        self.mask_df=mask_df = pd.read_pickle(self.data_path)


        train,validation=train_test_split(mask_df,test_size=0.3, random_state=0,stratify=mask_df['target'])

        self.train_df=MaskDataset(train)
        self.validation_df = MaskDataset(validation)

        mask_target=mask_df[mask_df['target']==1].shape[0]
        no_mask_target = mask_df[mask_df['target'] == 0].shape[0]

        all_targets = [ no_mask_target, mask_target]
        normed_weights= [1-(x/sum(all_targets)) for x in all_targets]
        self.crossEntrophyloss=CrossEntropyLoss(weight=torch.tensor(normed_weights))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_df, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validation_df, batch_size=32,  num_workers=4)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(),lr=self.learning_rate)

    def training_step(self, batch:dict, batch_idx:int) -> Dict[str,Tensor]:
        inputs,labels=batch['image'],batch['target']
        labels=labels.flatten()
        outputs=self.forward(inputs)
        loss=self.crossEntrophyloss(outputs,labels)

        tensorboardLogs={'train_loss':loss}
        return {'loss':loss, 'log':tensorboardLogs}

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Tensor]:
        inputs, labels = batch['image'], batch['target']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntrophyloss(outputs, labels)


        _,outputs=torch.max(outputs,dim=1)
        validation_acc= accuracy_score(outputs.cpu(),labels.cpu())
        validation_acc =torch.tensor(validation_acc)

        return {'val_loss': loss, 'val_acc': validation_acc}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) \
            -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        avgLoss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avgAcc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboardLogs = {'val_loss': avgLoss, 'val_acc':avgAcc}
        return {'val_loss': avgLoss, 'log': tensorboardLogs}

if __name__== '__main__':
    model=MaskDetector(Path('C:/Users/kamil/Desktop/mask_detector_pytorch/masks_df.pkl'))
    logger = TensorBoardLogger('logs/{}', name='my_model')

    checkpoint_callback=ModelCheckpoint(
        filepath="mask_detector_weights.ckpt",
        save_weights_only=True,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )

    trainer=Trainer(gpus=1 if torch.cuda.is_available() else 0,
                    max_epochs=10,
                    logger=[logger],
                    checkpoint_callback=checkpoint_callback,
                    profiler=True
                    )

    trainer.fit(model)



