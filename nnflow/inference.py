import time

import torch
from torch.nn import DataParallel

import numpy as np

from ezflow.utils import AverageMeter, InputPadder


def endpointerror(pred, target):
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

    if target.shape[1] == 3:
        """Ignore valid mask for EPE calculation."""
        target = target[:, :2, :, :]

    epe = torch.norm(pred - target, p=2, dim=1)

    metrics = {
        "epe": epe.mean().item()
    }

    return metrics


def warmup(model, dataloader, device, pad_divisor=1):
    """Performs an iteration of dataloading and model prediction to warm up CUDA device

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    pad_divisor : int, optional
        The divisor to make the image dimensions evenly divisible by using padding, by default 1
    """

    inp, target = next(iter(dataloader))
    img1, img2 = inp

    padder = InputPadder(img1.shape, divisor=pad_divisor)
    # print(img1.shape, img2.shape)
    img1, img2 = padder.pad(img1, img2)
    # print(img1.shape, img2.shape)

    img1, img2, target = (
        img1.to(device),
        img2.to(device),
        target.to(device),
    )

    _ = model(img1, img2)


def run_inference(model, dataloader, device, metric_fn, flow_scale=1.0, pad_divisor=1):
    """
    Uses a model to perform inference on a dataloader and captures inference time and evaluation metric

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    metric_fn : function
        Function to be used to calculate the evaluation metric
    flow_scale : float, optional
        Scale factor to be applied to the predicted flow
    pad_divisor : int, optional
        The divisor to make the image dimensions evenly divisible by using padding, by default 1

    Returns
    -------
    metric_meter : AverageMeter
        AverageMeter object containing the evaluation metric information
    avg_inference_time : float
        Average inference time

    """

    metric_meter = AverageMeter()
    times = []

    inp, target = next(iter(dataloader))
    batch_size = target.shape[0]

    padder = InputPadder(inp[0].shape, divisor=pad_divisor)

    f1_list = []

    with torch.no_grad():

        for inp, target in dataloader:

            img1, img2 = inp
            img1, img2, target = (
                img1.to(device),
                img2.to(device),
                target.to(device),
            )

            # print(img1.shape)
            img1, img2 = padder.pad(img1, img2)
            # print(img1.shape)

            torch.cuda.synchronize()

            start_time = time.time()

            output = model(img1, img2)

            torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

            pred = padder.unpad(output["flow_upsampled"])
            flow = pred * flow_scale

            metric = metric_fn(flow, target)
            metric_meter.update(metric["epe"], n=batch_size)

            if "f1" in metric:
                f1_list.append(metric["f1"])

    avg_inference_time = sum(times) / len(times)
    avg_inference_time /= batch_size  # Average inference time per sample

    fps = None
    f1_all = None

    if avg_inference_time != 0:
        fps =  1/avg_inference_time

        # print(
        #     f"Average inference time: {avg_inference_time}, FPS: {1/avg_inference_time}"
        # )

    if len(f1_list) > 0:
        f1_list = np.concatenate(f1_list)
        f1_all = 100 * np.mean(f1_list)

    metrics = {
        "avg_epe":metric_meter.avg,
        "avg_inference_time":avg_inference_time,
        "FPS": fps,
        "f1_all": f1_all
    }
    return metrics


def eval_model(
    model,
    dataloader,
    device,
    metric=endpointerror,
    profiler=None,
    flow_scale=1.0,
    pad_divisor=1
):
    """
    Evaluates a model on a dataloader and optionally profiles model characteristics such as memory usage, inference time, and evaluation metric

    Parameters
    ----------
    model : torch.nn.Module
        Model to be used for prediction / inference
    dataloader : torch.utils.data.DataLoader
        Dataloader to be used for prediction / inference
    device : torch.device
        Device (CUDA / CPU) to be used for prediction / inference
    metric : function, optional
        Function to be used to calculate the evaluation metric
    profiler : torch.profiler.profile, optional
        Profiler to be used for profiling model characteristics
    flow_scale : float, optional
        Scale factor to be applied to the predicted flow
    pad_divisor : int, optional
        The divisor to make the image dimensions evenly divisible by using padding, by default 1

    Returns
    -------
    float
        Average evaluation metric
    """

    if isinstance(device, list) or isinstance(device, tuple):
        device = ",".join(map(str, device))

    if device == "-1" or device == -1 or device == "cpu":
        device = torch.device("cpu")
        print("Running on CPU\n")

    elif not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA device(s) not available. Running on CPU\n")

    else:
        if device == "all":
            device = torch.device("cuda")
            model = DataParallel(model)
            print(f"Running on all available CUDA devices\n")

        else:
            if type(device) != str:
                device = str(device)

            device_ids = device.split(",")
            device_ids = [int(id) for id in device_ids]
            device = torch.device("cuda")

            model = DataParallel(model, device_ids=device_ids)
            print(f"Running on CUDA devices {device_ids}\n")

    model = model.to(device)
    model.eval()

    metric_fn = metric

    warmup(model, dataloader, device, pad_divisor=pad_divisor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if profiler is None:
        metrics = run_inference(
            model,
            dataloader,
            device,
            metric_fn,
            flow_scale=flow_scale,
            pad_divisor=pad_divisor,
        )
    else:
        metric_meter, _ = profile_inference(
            model,
            dataloader,
            device,
            metric_fn,
            profiler,
            flow_scale=flow_scale,
            pad_divisor=pad_divisor,
        )

    avg_epe = metrics["avg_epe"]
    f1_all = metrics["f1_all"]
    avg_inference_time = metrics["avg_inference_time"]
    FPS = metrics["FPS"]

    print(f"Average EPE = {avg_epe}")
    print(f"F1 all = {f1_all}")
    print(f"Average Inference time = {avg_inference_time}")
    print(f"Average FPS = {FPS}")
    print("-"*100,"\n")

    del dataloader
