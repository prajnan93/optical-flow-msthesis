import argparse
import torch

from ezflow.data import DataloaderCreator
from ezflow.models import build_model
from nnflow import *


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
        "--dataset", type=str, required=True, help="Name of the dataset"
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
        "--batch_size", type=int, default=8, help="Evaluate batch size"
    )
    parser.add_argument(
        "--pad_divisor", type=int, default=1, help="Padding divisor for dataset"
    )

    args = parser.parse_args()

    norm_params = {"use":True, "mean":args.mean, "std":args.std}

    val_loader = DataloaderCreator(batch_size=args.batch_size, num_workers=1, pin_memory=True)
    if args.dataset.lower() == "flyingchairs":
        val_loader.add_FlyingChairs(
            root_dir="../../../Datasets/FlyingChairs_release/data",
            split="validation",
            crop=False,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
            norm_params=norm_params
        )

    if args.dataset.lower() == "things_clean":
        val_loader.add_FlyingThings3D(
            root_dir="../../../Datasets/SceneFlow/FlyingThings3D",
            dstype="frames_cleanpass",
            split="validation",
            crop=False,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
            norm_params=norm_params
        )

    if args.dataset.lower() == "things_final":
        val_loader.add_FlyingThings3D(
            root_dir="../../../Datasets/SceneFlow/FlyingThings3D",
            dstype="frames_finalpass",
            split="validation",
            crop=False,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
            norm_params=norm_params
        )

    if args.dataset.lower() == "sceneflow":
        val_loader.add_SceneFlow(
            root_dir="../../../Datasets/SceneFlow",
            crop=False,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
            norm_params=norm_params
        )

    if args.dataset.lower() == "sintel_clean":
        val_loader.add_MPISintel(
            root_dir="../../../Datasets/MPI_Sintel/",
            split="training",
            dstype="clean",
            crop=False,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
            norm_params=norm_params
        )

    if args.dataset.lower() == "sintel_final":
        val_loader.add_MPISintel(
            root_dir="../../../Datasets/MPI_Sintel/",
            split="training",
            dstype="final",
            crop=False,
            crop_type="center",
            crop_size=args.crop_size,
            augment=False,
            norm_params=norm_params
        )

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

    eval_model(model, val_loader.get_dataloader(), device='0', pad_divisor=args.pad_divisor)


if __name__ == "__main__":

    main()