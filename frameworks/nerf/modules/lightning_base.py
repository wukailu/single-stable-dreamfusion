from abc import ABC, abstractmethod
from copy import deepcopy

import torch.nn
import torchmetrics
from pytorch_lightning import LightningModule


class NeRFModule(LightningModule, ABC):
    def __init__(self, hparams):
        super().__init__()  # must name after hparams or there will be plenty of bugs in early version of lightning
        self.save_hyperparameters(hparams)
        self.params = deepcopy(hparams)
        self.model_config_init(hparams)
        self.metrics = self.get_metrics()

    def get_metrics(self):
        return torch.nn.ModuleDict({"psnr": torchmetrics.image.PSNR()})

    def model_config_init(self, hparams):
        self.params = deepcopy(hparams)
        self.complete_hparams()
        self.steps_per_epoch = self.params['steps_per_epoch']

    def complete_hparams(self):
        default_list = {
            'optimizer': 'SGD',
            'lr_scheduler': 'ExpLR_step',
            'step_decay': 0.1 ** (1 / (20 * 1000)),
            'max_lr': 0.1,
            'weight_decay': 5e-4,
            'steps_per_epoch': 0,
        }
        self.params = {**default_list, **self.params}

    def choose_optimizer(self):
        params = self.parameters()
        from torch.optim import SGD, Adam
        if self.params['optimizer'] == 'SGD':
            optimizer = SGD(params, lr=self.params["max_lr"], weight_decay=self.params["weight_decay"], momentum=0.9,
                            nesterov=True)
        elif self.params['optimizer'] == 'Adam':
            optimizer = Adam(params, lr=self.params['max_lr'], weight_decay=self.params['weight_decay'])
        else:
            assert False, "optimizer not implemented"
        return optimizer

    def choose_scheduler(self, optimizer):
        if optimizer is None:
            return None

        from torch.optim import lr_scheduler
        if self.params['lr_scheduler'] == 'ExpLR_step':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.params['step_decay'])
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.params['lr_scheduler'] == 'StepLR_step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.params['decay_steps'],
                                            gamma=self.params['step_decay'])
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.params['lr_scheduler'] == 'StepAutoLR_step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=int(self.steps_per_epoch*0.8), gamma=0.1)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        elif self.params['lr_scheduler'] == 'OneCycLR':
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.params["max_lr"],
                                                steps_per_epoch=self.steps_per_epoch + 1,
                                                epochs=self.params["num_epochs"])
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
        else:
            print('lr_scheduler not found!')
            return None
        return scheduler

    def configure_optimizers(self):
        optimizer = self.choose_optimizer()
        scheduler = self.choose_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        else:
            return [optimizer], [scheduler]

    @abstractmethod
    def compute_loss(self, pred, labels):
        pass

    def step(self, batch, phase: str):
        images, labels = batch
        predictions = self.forward(images)
        loss = self.compute_loss(predictions, labels)
        for key, metric in self.metrics.items():
            self.log(phase + '/' + key, metric(predictions, labels))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'validation')

    def test_step(self, batch, batch_nb):
        return self.step(batch, 'test')