import random
import sys
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from ezflow.functional import FlowAugmentor, Normalize, crop
from ezflow.data import DataloaderCreator

class Kubric(data.Dataset):
    """
    Dataset class for preparing Kubric movi-e dataset for optical flow training.

    Parameters
    ----------
    root_dir : str
        path of the root directory for the flying chairs dataset
    split : str, default : "training"
        specify the training or validation split
    init_seed : bool, default : False
        If True, sets random seed to the worker
    is_prediction : bool, default : False
        If True,   If True, only image data are loaded for prediction otherwise both images and flow data are loaded
    append_valid_mask : bool, default :  False
        If True, appends the valid flow mask to the original flow mask at dim=0
    crop: bool, default : True
        Whether to perform cropping
    crop_size : :obj:`tuple` of :obj:`int`
        The size of the image crop
    crop_type : :obj:`str`, default : 'center'
        The type of croppping to be performed, one of "center", "random"
    augment : bool, default : False
        If True, applies data augmentation
    aug_params : :obj:`dict`
        The parameters for data augmentation
    norm_params : :obj:`dict`, optional
        The parameters for normalization
    """

    def __init__(
        self,
        root_dir,
        ds_type="movi_f",
        split="training",
        init_seed=False,
        is_prediction=False,
        append_valid_mask=False,
        crop=False,
        crop_size=(256, 256),
        crop_type="center",
        augment=True,
        aug_params={
            "color_aug_params": {"aug_prob": 0.2},
            "eraser_aug_params": {"aug_prob": 0.5},
            "spatial_aug_params": {"aug_prob": 0.8},
            "translate_params": {"aug_prob": 0.0},
            "rotate_params": {"aug_prob": 0.0},
        },
        sparse_transform=False,
        norm_params={"use": False},
    ):

        self.is_prediction = is_prediction
        self.init_seed = init_seed
        self.append_valid_mask = append_valid_mask
        self.crop = crop
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.sparse_transform = sparse_transform

        self.augment = augment
        self.augmentor = None

        self.normalize = Normalize(**norm_params)

        # Tensorflow prioritizes loading on GPU by default
        # Disable loading on all GPUS
        tf.config.set_visible_devices([], 'GPU')
        
        split = "train" if split == "training" else "validation"
        try:
            self.ds = tfds.load(ds_type, data_dir=root_dir, split=split, shuffle_files=True)
            
            self.sample_count = self.ds.cardinality().numpy() * 24
            self.ds_iterator = iter(self.ds)
        except:
            print(f"Kubric Dataset {ds_type} not found in location {ds_path}")
            sys.exit()
        
        if augment:
            self.augmentor = FlowAugmentor(crop_size=crop_size, **aug_params)
        

    def __getitem__(self, index):
        """
        Returns the corresponding images and the flow between them.

        Parameters
        ----------
        index : int
            specify the index location for access to Dataset item

        Returns
        -------
        tuple
            A tuple consisting of ((img1, img2), flow)

            img1 and img2 of shape 3 x H x W.
            flow of shape 2 x H x W if append_valid_mask is False.
            flow of shape 3 x H x W if append_valid_mask is True.
        """

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        
        try:
            sample = next(self.ds_iterator)
        except:
            self.ds_iterator = iter(self.ds)
            sample = next(self.ds_iterator)
            
        video = sample['video'].numpy()
        flows = sample['forward_flow'].numpy()
        
        max_index = video.shape[0] - 2
        random_index = np.random.randint(0, max_index)
        
        img1, img2 = video[random_index].copy(), video[random_index+1].copy()
        flow = flows[random_index].copy()
        
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        flow = flow.astype(np.float32)
        valid = None
        
        # delete unused variables to free memory
        del video, flows, sample, max_index, random_index
        
        if self.is_prediction:

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            img1, img2 = self.normalize(img1, img2)
            return img1, img2

        if self.augment is True and self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        if self.crop is True:
            img1, img2, flow, valid = crop(
                img1,
                img2,
                flow,
                valid=valid,
                crop_size=self.crop_size,
                crop_type=self.crop_type,
                sparse_transform=self.sparse_transform,
            )

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        
        img1, img2 = self.normalize(img1, img2)

        if self.append_valid_mask:
            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

            valid = valid.float()
            valid = torch.unsqueeze(valid, dim=0)
            flow = torch.cat([flow, valid], dim=0)

        return (img1, img2), flow

    def __len__(self):
        """
        Return length of the dataset.

        """
        return self.sample_count


class CustomDataloaderCreator(DataloaderCreator):

    def __init__(
        self,
        batch_size,
        pin_memory=False,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        init_seed=False,
        append_valid_mask=False,
        is_prediction=False,
        distributed=False,
        world_size=None,
    ):

        super(CustomDataloaderCreator, self).__init__(
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            init_seed=init_seed,
            append_valid_mask=append_valid_mask,
            is_prediction=is_prediction,
            distributed=distributed,
            world_size=world_size,
        )


    def add_Kubric(self, root_dir, split="training", augment=False, **kwargs):
        self.dataset_list.append(
            Kubric(
                root_dir,
                split=split,
                init_seed=self.init_seed,
                is_prediction=self.is_prediction,
                append_valid_mask=self.append_valid_mask,
                augment=augment,
                **kwargs,
            )
        )