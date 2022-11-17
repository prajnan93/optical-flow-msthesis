import argparse
import torch

import numpy as np

from ezflow.data import DataloaderCreator
from ezflow.models import build_model
from nnflow import *
from nnflow.inference import endpointerror

import warnings
warnings.filterwarnings("ignore")

def epe_f1_metric(pred, target):
    """
    Endpoint error
    Parameters
    ----------
    pred : torch.Tensor
        Predicted flow
    target : torch.Tensor
        Target flow
    Returns
    -------
    torch.Tensor
        Endpoint error
    """
    if isinstance(pred, tuple) or isinstance(pred, list):
        pred = pred[-1]

    valid_mask = None
    if target.shape[1] == 3:
        valid_mask = target[:, 2:, :, :]
        target = target[:, :2, :, :]

    epe = torch.norm(pred - target, p=2, dim=1)
    f1 = None

    if valid_mask is not None:
        mag = torch.sum(target**2, dim=1).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_mask.reshape(-1) >= 0.5

        f1 = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        
        epe = epe[val].mean().item()
        f1 = f1[val].cpu().numpy()

    else:
        epe = epe.mean().item()
    

    metrics = {
        "epe": epe,
        "f1": f1
    }

    return metrics

def main():

    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--model_cfg", type=str, required=True, help="Path to the model config file"
    )
    parser.add_argument(
        "--model_weights_path", type=str, required=True, help="Path of the model weights"
    )
    parser.add_argument(
        "--dataset", required=True, type=str, help="Name of the dataset"
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=[368,496],
        required=False,
        help="Crop size for validation images",
    )
    parser.add_argument(
        "--mean",
        type=float,
        nargs="+",
        default=None,
        required=True,
        help="mean for normalization",
    )
    parser.add_argument(
        "--std",
        type=float,
        nargs="+",
        default=None,
        required=True,
        help="standard deviation for normalization",
    )
    parser.add_argument(
        "--flow_scale", type=float, required=True, default=1.0, help="Target Flow Scale Factor"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Evaluate batch size"
    )
    parser.add_argument(
        "--pad_divisor", type=int, default=1, help="Padding divisor for dataset"
    )
    parser.add_argument(
        "--raft_iters",
        type=int,
        default=None,
        help="Number of RAFT iters",
    )

    args = parser.parse_args()

    norm_params = {"use":True, "mean":args.mean, "std":args.std}

    loaders = {}


    print(f"Evaluating checkpoint {args.model_weights_path}")
    
    ds_list = args.dataset.lower().split()

    if 'chairs' in ds_list:
        chair_loader = DataloaderCreator(batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        chair_loader.add_FlyingChairs(
                root_dir="../../../Datasets/FlyingChairs_release/data",
                split="validation",
                crop=False,
                crop_type="center",
                crop_size=args.crop_size,
                augment=False,
                norm_params=norm_params
            )

        loaders['chairs'] = {
            'loader': chair_loader,
            'pad_div': {
                'RAFT':1,
                'PWCNet':1,
                'FlowNetC':1,
                'GMFlowV2':1
            }
        }

    if 'sintel' in ds_list:
        sintel_clean_loader = DataloaderCreator(batch_size=args.batch_size,  shuffle=False, num_workers=4, pin_memory=True)
        sintel_clean_loader.add_MPISintel(
                root_dir="../../../Datasets/MPI_Sintel/",
                split="training",
                dstype="clean",
                crop=False,
                crop_type="center",
                crop_size=args.crop_size,
                augment=False,
                norm_params=norm_params
            )


        sintel_final_loader = DataloaderCreator(batch_size=args.batch_size,  shuffle=False, num_workers=4, pin_memory=True)
        sintel_final_loader.add_MPISintel(
                root_dir="../../../Datasets/MPI_Sintel/",
                split="training",
                dstype="final",
                crop=False,
                crop_type="center",
                crop_size=args.crop_size,
                augment=False,
                norm_params=norm_params
            )

        loaders['sintel_clean'] = {
            'loader': sintel_clean_loader,
            'pad_div': {
                'RAFT':8,
                'PWCNet':16,
                'FlowNetC':16,
                'GMFlowV2':16
            }
        }
        
        loaders['sintel_final'] = {
            'loader': sintel_final_loader,
            'pad_div': {
                'RAFT':8,
                'PWCNet':16,
                'FlowNetC':16,
                'GMFlowV2':16
            }
        }
        
    if 'kitti' in ds_list:
        kitti_loader = DataloaderCreator(batch_size=args.batch_size,  shuffle=False, append_valid_mask=True, num_workers=4, pin_memory=True)
        kitti_loader.add_Kitti(
                root_dir="../../../Datasets/KITTI2015/",
                split="training",
                crop=True,
                crop_type="center",
                crop_size=[368, 1216], # 368, 1216, #370, 1224
                augment=False,
                norm_params=norm_params
            )
        loaders['kitti'] = {
            'loader': kitti_loader,
            'pad_div': {
                'RAFT':8,
                'PWCNet':32,
                'FlowNetC':32,
                'GMFlowV2':8
            }
        }
        


    # if 'things_clean' in ds_list:
    #     val_loader = DataloaderCreator(batch_size=args.batch_size,  shuffle=False, num_workers=4, pin_memory=True)
    #     val_loader.add_FlyingThings3D(
    #         root_dir="../../../Datasets/SceneFlow/FlyingThings3D",
    #         dstype="frames_cleanpass",
    #         split="validation",
    #         crop=False,
    #         crop_type="center",
    #         crop_size=args.crop_size,
    #         augment=False,
    #         norm_params=norm_params
    #     )
    #     loaders["things_clean"] = val_loader

    # if 'things_final' in ds_list:
    #     val_loader = DataloaderCreator(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    #     val_loader.add_FlyingThings3D(
    #         root_dir="../../../Datasets/SceneFlow/FlyingThings3D",
    #         dstype="frames_finalpass",
    #         split="validation",
    #         crop=False,
    #         crop_type="center",
    #         crop_size=args.crop_size,
    #         augment=False,
    #         norm_params=norm_params
    #     )
    #     loaders["things_final"]=val_loader

    # if 'kubric' in ds_list:
    #     val_loader = CustomDataloaderCreator(batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    #     val_loader.add_Kubric(
    #         root_dir="../KubricFlow",
    #         split="validation",
    #         crop=False,
    #         crop_type="center",
    #         crop_size=args.crop_size,
    #         augment=False,
    #         norm_params=norm_params
    #     )
    #     loaders["kubric"] = val_loader


    model = build_model(
        args.model, 
        cfg_path=args.model_cfg, 
        custom_cfg=True
    )

    state_dict = torch.load(args.model_weights_path, map_location=torch.device('cpu'))
    if "model_state_dict" in state_dict:
        model_state_dict = state_dict["model_state_dict"]
    elif "model" in state_dict:
        model_state_dict = state_dict["model"]
    else:
        model_state_dict = state_dict

    model.load_state_dict(model_state_dict)

    if args.raft_iters is not None:
        model.cfg.UPDATE_ITERS = args.raft_iters
        print("RAFT UPDATE ITERS: ", model.cfg.UPDATE_ITERS)

    metric = endpointerror

    for name in loaders:
        print(f"Evaluating {name}:")
        loader = loaders[name]['loader'].get_dataloader()

        pad_divisor = loaders[name]['pad_div'][args.model]

        if name == "kitti":
            metric = epe_f1_metric

        eval_model(
            model, 
            loader, 
            metric=metric, 
            device='0', 
            pad_divisor=pad_divisor, 
            flow_scale=args.flow_scale
        )

    print("Evaluation completed!!")

if __name__ == "__main__":

    main()