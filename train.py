import argparse

from ezflow.data import DataloaderCreator
from ezflow.engine import Trainer, DistributedTrainer, get_training_cfg
from ezflow.models import build_model

from nnflow import *

def main():

    parser = argparse.ArgumentParser(description="Train a model")

    parser.add_argument(
        "--train_cfg", type=str, required=True, help="Path to the training config file"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train"
    )
    parser.add_argument(
        "--model_cfg", type=str, required=False, help="Path to the model config file"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory where logs are to be written",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Directory where ckpts are to be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to train for",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume training from a previous ckpt",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Path to ckpt for resuming training",
    )
    parser.add_argument(
        "--resume_epochs",
        type=int,
        default=None,
        help="Number of epochs to train after resumption",
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="0",
        help="Device ID"
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="nccl",
        help="Backend to use for distributed computing",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        help="World size for Distributed Training"
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        help="Enable mixed precision",
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        help="Enable scheduler",
    )
    parser.add_argument(
        "--freeze_batch_norm",
        action="store_true",
        help="Sets all batch norm layers to eval state.",
    )
    parser.add_argument(
        "--train_ds",
        type=str,
        help="Name of the dataset \n. Supported datasets: AutoFlow, FlyingChairs, FlyingThings3D, MPISintel, KITTI, SceneFlow",
    )
    parser.add_argument(
        "--train_ds_dir",
        type=str,
        help="Path of root directory for the training dataset",
    )
    parser.add_argument(
        "--val_ds",
        type=str,
        help="Name of the dataset \n. Supported datasets: AutoFlow, FlyingChairs, FlyingThings3D, MPISintel, KITTI, SceneFlow",
    )
    parser.add_argument(
        "--val_ds_dir",
        type=str,
        help="Path of root directory for the validation dataset",
    )
    parser.add_argument(
        "--train_crop_size",
        type=int,
        nargs="+",
        default=None,
        help="Crop size for training images",
    )
    parser.add_argument(
        "--val_crop_size",
        type=int,
        nargs="+",
        default=None,
        help="Crop size for validation images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, required=False, help="Learning rate")

    args = parser.parse_args()

    training_cfg = get_training_cfg(cfg_path=args.train_cfg)
    training_cfg.LOG_DIR = args.log_dir
    training_cfg.CKPT_DIR = args.ckpt_dir
    training_cfg.DATA.BATCH_SIZE = args.batch_size
    training_cfg.DEVICE = args.device
    training_cfg.DISTRIBUTED.BACKEND = args.distributed_backend
    training_cfg.MIXED_PRECISION = args.use_mixed_precision
    training_cfg.SCHEDULER.USE = args.use_scheduler
    training_cfg.FREEZE_BATCH_NORM = args.freeze_batch_norm

    if args.world_size is not None:
        training_cfg.DISTRIBUTED.WORLD_SIZE = args.world_size

    if args.train_ds is not None and args.train_ds_dir is not None:
        training_cfg.DATA.TRAIN_DATASET.NAME = args.train_ds
        training_cfg.DATA.TRAIN_DATASET.ROOT_DIR = args.train_ds_dir

    if args.val_ds is not None and args.val_ds_dir is not None:
        training_cfg.DATA.VAL_DATASET.NAME = args.val_ds
        training_cfg.DATA.VAL_DATASET.ROOT_DIR = args.val_ds_dir

    if args.train_crop_size is not None:
        training_cfg.DATA.TRAIN_CROP_SIZE = args.train_crop_size

    if args.val_crop_size is not None:
        training_cfg.DATA.VAL_CROP_SIZE = args.val_crop_size

    if args.lr is not None:
        training_cfg.OPTIMIZER.LR = args.lr

        if training_cfg.SCHEDULER.NAME == "OneCycleLR":
            training_cfg.SCHEDULER.PARAMS.max_lr = args.lr

    if args.num_steps is not None:
        training_cfg.NUM_STEPS = args.num_steps

        if training_cfg.SCHEDULER.NAME == "OneCycleLR":
            training_cfg.SCHEDULER.PARAMS.total_steps = args.num_steps + 100


    if args.epochs is not None:
        training_cfg.EPOCHS = args.epochs

        if training_cfg.SCHEDULER.NAME == "OneCycleLR":
            training_cfg.SCHEDULER.PARAMS.epochs = args.epochs

    if training_cfg.DISTRIBUTED.USE is True:

        train_loader_creator = DataloaderCreator(
            batch_size=training_cfg.DATA.BATCH_SIZE,
            num_workers=training_cfg.DATA.NUM_WORKERS,
            pin_memory=training_cfg.DATA.PIN_MEMORY,
            distributed=True,
            world_size=training_cfg.DISTRIBUTED.WORLD_SIZE,
            append_valid_mask=training_cfg.DATA.APPEND_VALID_MASK
        )

        val_loader_creator = DataloaderCreator(
            batch_size=training_cfg.DATA.BATCH_SIZE,
            num_workers=training_cfg.DATA.NUM_WORKERS,
            pin_memory=training_cfg.DATA.PIN_MEMORY,
            distributed=True,
            world_size=training_cfg.DISTRIBUTED.WORLD_SIZE,
            append_valid_mask=training_cfg.DATA.APPEND_VALID_MASK
        )

    else:

        train_loader_creator = DataloaderCreator(
            batch_size=training_cfg.DATA.BATCH_SIZE,
            num_workers=training_cfg.DATA.NUM_WORKERS,
            pin_memory=training_cfg.DATA.PIN_MEMORY,
            append_valid_mask=training_cfg.DATA.APPEND_VALID_MASK
        )

        val_loader_creator = DataloaderCreator(
            batch_size=training_cfg.DATA.BATCH_SIZE,
            num_workers=training_cfg.DATA.NUM_WORKERS,
            pin_memory=training_cfg.DATA.PIN_MEMORY,
            append_valid_mask=training_cfg.DATA.APPEND_VALID_MASK
        )


    # --------------------- TRAINING DATASETS -----------------------------------#
    if training_cfg.DATA.TRAIN_DATASET.NAME.lower() == "flyingchairs":
        train_loader_creator.add_FlyingChairs(
            root_dir=training_cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="random",
            crop_size=training_cfg.DATA.TRAIN_CROP_SIZE,
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
                "color_aug_params":training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.COLOR_AUG_PARAMS,
                "eraser_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.ERASER_AUG_PARAMS
            },
        )

    if training_cfg.DATA.TRAIN_DATASET.NAME.lower() == "flyingthings3d":
        train_loader_creator.add_FlyingThings3D(
            root_dir=training_cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            dstype="frames_cleanpass",
            crop=True,
            crop_type="random",
            crop_size=training_cfg.DATA.TRAIN_CROP_SIZE,
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )
        train_loader_creator.add_FlyingThings3D(
            root_dir=training_cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            dstype="frames_finalpass",
            crop=True,
            crop_type="random",
            crop_size=training_cfg.DATA.TRAIN_CROP_SIZE,
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )

    if training_cfg.DATA.TRAIN_DATASET.NAME.lower() == "sceneflow":
        train_loader_creator.add_SceneFlow(
            root_dir=training_cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="random",
            crop_size=training_cfg.DATA.TRAIN_CROP_SIZE,
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )

    if training_cfg.DATA.TRAIN_DATASET.NAME.lower() == "mpisintel":
        train_loader_creator.add_MPISintel(
            root_dir=training_cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="random",
            crop_size=training_cfg.DATA.TRAIN_CROP_SIZE,
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )

    if training_cfg.DATA.TRAIN_DATASET.NAME.lower() == "kitti":
        train_loader_creator.add_Kitti(
            root_dir=training_cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="random",
            crop_size=training_cfg.DATA.TRAIN_CROP_SIZE,
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )

    if training_cfg.DATA.TRAIN_DATASET.NAME.lower() == "autoflow":
        train_loader_creator.add_AutoFlow(
            root_dir=training_cfg.DATA.TRAIN_DATASET.ROOT_DIR,
            crop=True,
            crop_type="random",
            crop_size=training_cfg.DATA.TRAIN_CROP_SIZE,
            augment=training_cfg.DATA.AUGMENTATION.USE,
            aug_params={
                "spatial_aug_params": training_cfg.DATA.AUGMENTATION.PARAMS.TRAINING.SPATIAL_AUG_PARAMS,
            },
        )

    # --------------------- VALIDATION DATASETS -----------------------------------#
    if training_cfg.DATA.VAL_DATASET.NAME.lower() == "flyingchairs":
        val_loader_creator.add_FlyingChairs(
            root_dir=training_cfg.DATA.VAL_DATASET.ROOT_DIR,
            split="validation",
            crop=True,
            crop_type="center",
            crop_size=training_cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if training_cfg.DATA.VAL_DATASET.NAME.lower() == "flyingthings3d":
        val_loader_creator.add_FlyingThings3D(
            root_dir=training_cfg.DATA.VAL_DATASET.ROOT_DIR,
            dstype="frames_cleanpass",
            split="validation",
            crop=True,
            crop_type="center",
            crop_size=training_cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )
        val_loader_creator.add_FlyingThings3D(
            root_dir=training_cfg.DATA.VAL_DATASET.ROOT_DIR,
            dstype="frames_finalpass",
            split="validation",
            crop=True,
            crop_type="center",
            crop_size=training_cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if training_cfg.DATA.VAL_DATASET.NAME.lower() == "sceneflow":
        val_loader_creator.add_SceneFlow(
            root_dir=training_cfg.DATA.VAL_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=training_cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if training_cfg.DATA.VAL_DATASET.NAME.lower() == "mpisintel":
        val_loader_creator.add_MPISintel(
            root_dir=training_cfg.DATA.VAL_DATASET.ROOT_DIR,
            split="training",
            dstype="clean",
            crop=True,
            crop_type="center",
            crop_size=training_cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if training_cfg.DATA.VAL_DATASET.NAME.lower() == "kitti":
        val_loader_creator.add_Kitti(
            root_dir=training_cfg.DATA.VAL_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=training_cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if training_cfg.DATA.VAL_DATASET.NAME.lower() == "autoflow":
        val_loader_creator.add_AutoFlow(
            root_dir=training_cfg.DATA.VAL_DATASET.ROOT_DIR,
            crop=True,
            crop_type="center",
            crop_size=training_cfg.DATA.VAL_CROP_SIZE,
            augment=False,
        )

    if args.model_cfg is not None:
        model = build_model(args.model, cfg_path=args.model_cfg, custom_cfg=True)
    else:
        model = build_model(args.model, default=True)

    

    if training_cfg.DISTRIBUTED.USE is True:
        trainer = DistributedTrainer(
            training_cfg, 
            model, 
            train_loader_creator = train_loader_creator, 
            val_loader_creator = val_loader_creator
        )

        if args.resume:
            assert (
                args.resume_ckpt is not None
            ), "Please provide a ckpt to resume training from"
            print("Resuming distributed training")
            trainer.resume_training(args.resume_ckpt)

        else:
            print("Distributed training")
            trainer.train()

    else:
        trainer = Trainer(
            training_cfg, 
            model, 
            train_loader = train_loader_creator.get_dataloader(), 
            val_loader = val_loader_creator.get_dataloader()
        )
        
        if args.resume:
            assert (
                args.resume_ckpt is not None
            ), "Please provide a ckpt to resume training from"
            print("Resuming training")
            trainer.resume_training(args.resume_ckpt)

        else:
            print("Training")
            trainer.train()


if __name__ == "__main__":

    main()
