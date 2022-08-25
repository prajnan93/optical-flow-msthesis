import os
import time
from copy import deepcopy
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from ezflow.engine import Trainer, DistributedTrainer
from ezflow.utils import AverageMeter


class CustomTrainer(Trainer):
    
    def __init__(self, cfg, model, train_loader, val_loader):
        super(CustomTrainer, self).__init__(cfg, model, train_loader, val_loader)

    def _validate_model(self, iter_type, iterations):
        self.model.eval()
        metric_meter = AverageMeter()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for inp, target in self.val_loader:
                img1, img2 = inp
                img1, img2, target = (
                    img1.to(self.device),
                    img2.to(self.device),
                    target.to(self.device),
                )

                flow_up, flow_preds = self.model(img1, img2)
                loss = self.loss_fn(flow_preds, target / self.cfg.TARGET_SCALE_FACTOR)
                loss_meter.update(loss.item())
                metric = self._calculate_metric(flow_up, target)
                metric_meter.update(metric)

        new_avg_val_loss, new_avg_val_metric = loss_meter.avg, metric_meter.avg

        print("\n", "-" * 80)
        self.writer.add_scalar("avg_validation_loss", new_avg_val_loss, iterations)
        print(
            f"\n{iter_type} {iterations}: Average validation loss = {new_avg_val_loss}"
        )

        self.writer.add_scalar("avg_validation_metric", new_avg_val_metric, iterations)
        print(
            f"{iter_type} {iterations}: Average validation metric = {new_avg_val_metric}\n"
        )
        print("-" * 80, "\n")

        self._save_best_model(new_avg_val_loss, new_avg_val_metric)

        self.model.train()
        self._freeze_bn()

class CustomDistributedTrainer(DistributedTrainer):
    
    def __init__(self, cfg, model, train_loader_creator, val_loader_creator):
        super(CustomDistributedTrainer, self).__init__(cfg, model, train_loader_creator, val_loader_creator)

    def _validate_model(self, iter_type, iterations):
        self.model.eval()
        metric_meter = AverageMeter()
        loss_meter = AverageMeter()

        with torch.no_grad():
            for inp, target in self.val_loader:
                img1, img2 = inp
                img1, img2, target = (
                    img1.to(self.device),
                    img2.to(self.device),
                    target.to(self.device),
                )

                flow_up, flow_preds = self.model(img1, img2)
                loss = self.loss_fn(flow_preds, target / self.cfg.TARGET_SCALE_FACTOR)
                loss_meter.update(loss.item())
                metric = self._calculate_metric(flow_up, target)
                metric_meter.update(metric)

        new_avg_val_loss, new_avg_val_metric = loss_meter.avg, metric_meter.avg

        print("\n", "-" * 80)
        self.writer.add_scalar("avg_validation_loss", new_avg_val_loss, iterations)
        print(
            f"\n{iter_type} {iterations}: Average validation loss = {new_avg_val_loss}"
        )

        self.writer.add_scalar("avg_validation_metric", new_avg_val_metric, iterations)
        print(
            f"{iter_type} {iterations}: Average validation metric = {new_avg_val_metric}\n"
        )
        print("-" * 80, "\n")

        self._save_best_model(new_avg_val_loss, new_avg_val_metric)

        self.model.train()
        self._freeze_bn()