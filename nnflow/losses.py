from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ezflow.functional import FUNCTIONAL_REGISTRY
from ezflow.config import configurable

def endpoint_error(pred: torch.Tensor,
                   target: torch.Tensor,
                   p: int = 2,
                   q: Optional[float] = None,
                   eps: Optional[float] = None) -> torch.Tensor:
    r"""Calculate end point errors between prediction and ground truth.
    If not define q, the loss function is
    .. math::
      loss = \Vert \mathbf{u}-\mathbf{u_gt} \Vert^p
    otherwise,
    .. math::
      loss = (\Vert \mathbf{u}-\mathbf{u_gt} \Vert^p+eps)^q
    For PWC-Net L2 norm loss: p=2, for the robust loss function p=1, q=0.4,
    eps=0.01.
    Args:
        pred (torch.Tensor): output flow map from flow_estimator
            shape(B, 2, H, W).
        target (torch.Tensor): ground truth flow map shape(B, 2, H, W).
        p (int): norm degree for loss. Options are 1 or 2. Defaults to 2.
        q (float, optional): used to give less penalty to outliers when
            fine-tuning model. Defaults to 0.4.
        eps (float, optional): a small constant to numerical stability when
            fine-tuning model. Defaults to 0.01.
    Returns:
        Tensor: end-point error map with the shape (B, H, W).
    """

    assert pred.shape == target.shape, \
        (f'pred shape {pred.shape} does not match target '
         f'shape {target.shape}.')

    epe_map = torch.norm(pred - target, p, dim=1)  # shape (B, H, W).

    if q is not None and eps is not None:
        epe_map = (epe_map + eps)**q

    return epe_map


def multi_level_flow_loss(loss_function,
                          preds_dict: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
                          target: torch.Tensor,
                          weights: Dict[str, float] = dict(
                              level6=0.32,
                              level5=0.08,
                              level4=0.02,
                              level3=0.01,
                              level2=0.005),
                          valid: Optional[torch.Tensor] = None,
                          flow_div: float = 20.,
                          max_flow: float = float('inf'),
                          resize_flow: str = 'downsample',
                          reduction: str = 'sum',
                          scale_as_level: bool = False,
                          **kwargs) -> torch.Tensor:
    """Multi-level endpoint error loss function.
    Args:
        loss_function: pixel-wise loss function for optical flow map.
        preds_dict (dict): multi-level output of predicted optical flow, and
            the contain in dict is a Tensor or list of Tensor with shape
            (B, 1, H_l, W_l), where l indicates the level.
        target (Tensor): ground truth of optical flow with shape (B, 2, H, W).
        weights (dict): manual rescaling weights given to the loss of flow map
            at each level, and the keys in weights must correspond to predicted
            dict. Defaults to dict(
            level6=0.32, level5=0.08, level4=0.02, level3=0.01, level2=0.005).
        valid (Tensor, optional): valid mask for optical flow.
            Defaults to None.
        flow_div (float): the divisor used to scale down ground truth.
            Defaults to 20.
        max_flow (float): maximum value of optical flow, if some pixel's flow
            of target is larger than it, this pixel is not valid. Default to
            float('inf').
        reduction (str): the reduction to apply to the output:'none', 'mean',
            'sum'. 'none': no reduction will be applied and will return a
            full-size epe map, 'mean': the mean of the epe map is taken, 'sum':
            the epe map will be summed but averaged by batch_size.
            Default: 'sum'.
        resize_flow (str): mode for reszing flow: 'downsample' and 'upsample',
            as multi-level predicted outputs don't match the ground truth.
            If set to 'downsample', it will downsample the ground truth, and
            if set to 'upsample' it will upsample the predicted flow, and
            'upsample' is used for sparse flow map as no generic interpolation
            mode can resize a ground truth of sparse flow correctly.
            Default to 'downsample'.
        scale_as_level (bool): Whether flow for each level is at its native
            spatial resolution. If `'scale_as_level'` is True, the ground
            truth is scaled at different levels, if it is False, the ground
            truth will not be scaled. Default to False.
        kwargs: arguments for loss_function.
    Returns:
        Tensor: end-point error loss.
    """

    assert isinstance(weights, dict)

    assert list(preds_dict.keys()).sort() == list(weights.keys()).sort(), \
        'Error: Keys of prediction do not match keys of weights!'

    mag = torch.sum(target**2, dim=1).sqrt()

    if valid is None:
        valid = torch.ones_like(target[:, 0, :, :])
    else:
        valid = ((valid >= 0.5) & (mag < max_flow)).to(target)

    target_div = target / flow_div

    c_org, h_org, w_org = target.shape[1:]
    assert c_org == 2, f'The channels ground truth must be 2, but got {c_org}'

    loss = 0

    for level in weights.keys():

        #print(f"level: {level} target : {target_div.shape}")
        # predict more than one flow map at one level
        cur_pred = preds_dict[level] if isinstance(
            preds_dict[level], (tuple, list)) else [preds_dict[level]]

        num_preds = len(cur_pred)

        b, _, h, w = cur_pred[0].shape

        scale_factor = torch.Tensor([
            float(w / w_org), float(h / h_org)
        ]).to(target) if scale_as_level else torch.Tensor([1., 1.]).to(target)

        cur_weight = weights.get(level)

        if resize_flow == 'downsample':
            # down sample ground truth following irr solution
            # https://github.com/visinf/irr/blob/master/losses.py#L16
            cur_target = F.adaptive_avg_pool2d(target_div, [h, w])
            cur_valid = F.adaptive_max_pool2d(valid, [h, w])
        else:
            cur_target = target_div
            cur_valid = valid

        loss_map = torch.zeros_like(cur_target[:, 0, ...])
        #print(f"scale factor : {scale_factor.shape} downsampled target : {cur_target.shape}")
        #print()

        for i_pred in cur_pred:

            if resize_flow == 'upsample':
                # up sample predicted flow following pwcnet and irr solution
                # https://github.com/visinf/irr/blob/master/losses.py#L20
                # when training sparse flow dataset, as no generic
                # interpolation mode can resize a ground truth of sparse flow
                # correctly.
                i_pred = F.interpolate(
                    i_pred,
                    size=cur_target.shape[2:],
                    mode='bilinear',
                    align_corners=False)

            cur_target = torch.einsum('b c h w, c -> b c h w', cur_target,
                                      scale_factor)

            #print(f"pred: {i_pred.shape} target: {cur_target.shape} loss map: {loss_map.shape} weight: {cur_weight}")
            loss_map += loss_function(i_pred, cur_target, **kwargs) * cur_valid

            if reduction == 'mean':
                loss += loss_map.sum() / (cur_valid.sum() + 1e-8) * cur_weight
            elif reduction == 'sum':
                loss += loss_map.sum() / b * cur_weight

        #print(40*'-')

    #print(f"total preds: {num_preds}")
    return loss / num_preds


@FUNCTIONAL_REGISTRY.register()
class MultiLevelEPE(nn.Module):
    """Multi-level end point error loss.
    Args:
        p (int): norm degree for loss. Options are 1 or 2. Defaults to 2.
        q (float): used to give less penalty to outliers when fine-tuning
            model. Defaults to None.
        eps (float): a small constant to numerical stability when fine-tuning
            model. Defaults to None.
        weights (dict): manual rescaling weights given to the loss of flow map
            at each level, and the keys in weights must correspond to predicted
            dict. Defaults to dict(
            level6=0.32, level5=0.08, level4=0.02, level3=0.01, level2=0.005).
        flow_div (float): the divisor used to scale down ground truth.
            Defaults to 20.
        max_flow (float): maximum value of optical flow, if some pixel's flow
            of target is larger than it, this pixel is not valid. Default to
            float('inf').
        resize_flow (str): mode for reszing flow: 'downsample' and 'upsample',
            as multi-level predicted outputs don't match the ground truth.
            If set to 'downsample', it will downsample the ground truth, and
            if set to 'upsample' it will upsample the predicted flow, and
            'upsample' is used for sparse flow map as no generic interpolation
            mode can resize a ground truth of sparse flow correctly.
            Default to 'downsample'.
        scale_as_level (bool): Whether flow for each level is at its native
            spatial resolution. If `'scale_as_level'` is True, the ground
            truth is scaled at different levels, if it is False, the ground
            truth will not be scaled. Default to False.
        reduction (str): the reduction to apply to the output:'none', 'mean',
            'sum'. 'none': no reduction will be applied and will return a
            full-size epe map, 'mean': the mean of the epe map is taken, 'sum':
            the epe map will be summed but averaged by batch_size.
            Default: 'sum'.
    """

    def __init__(self,
                 p: int = 2,
                 q: Optional[float] = None,
                 eps: Optional[float] = None,
                 weights: Dict[str, float] = dict(
                     level6=0.32,
                     level5=0.08,
                     level4=0.02,
                     level3=0.01,
                     level2=0.005),
                 flow_div: float = 20.,
                 max_flow: float = float('inf'),
                 resize_flow: str = 'downsample',
                 scale_as_level: bool = False,
                 reduction: str = 'sum') -> None:

        super().__init__()

        assert p == 1 or p == 2
        self.p = p

        self.q = q
        if self.q is not None:
            assert self.q > 0

        self.eps = eps
        if self.eps is not None:
            assert eps > 0

        assert flow_div > 0
        self.flow_div = flow_div

        assert isinstance(weights, dict)
        self.weights = weights

        assert max_flow > 0.
        self.max_flow = max_flow

        assert resize_flow in ('downsample', 'upsample')
        self.resize_flow = resize_flow

        assert isinstance(scale_as_level, bool)
        self.scale_as_level = scale_as_level

        assert reduction in ('mean', 'sum')
        self.reduction = reduction


    def forward(self,
                preds,
                target: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function for MultiLevelEPE.
        Args:
            preds_dict (dict): Multi-level output of predicted optical flow,
                and the contain in dict is a Tensor or list of Tensor with
                shape (B, 1, H_l, W_l), where l indicates the level.
            target (Tensor): Ground truth of optical flow with shape
                (B, 2, H, W).
            valid (Tensor, optional): Valid mask for optical flow.
                Defaults to None.
        Returns:
            Tensor: value of pixel-wise end point error loss.
        """

        if isinstance(preds, dict):
            preds_dict = preds
        else:
            preds_dict = {
                'level6': preds[4],
                'level5': preds[3],
                'level4': preds[2],
                'level3': preds[1],
                'level2': preds[0]
            }

        # for level in preds_dict:
        #     #print(f"{level} {preds_dict[level].shape}")

        return multi_level_flow_loss(
            endpoint_error,
            preds_dict,
            target,
            weights=self.weights,
            valid=valid,
            flow_div=self.flow_div,
            max_flow=self.max_flow,
            resize_flow=self.resize_flow,
            scale_as_level=self.scale_as_level,
            reduction=self.reduction,
            p=self.p,
            q=self.q,
            eps=self.eps,
        )

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue


@FUNCTIONAL_REGISTRY.register()
class MultiScale(nn.Module):
    def __init__(self, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L2'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.l_type = norm
        #self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE']

    def forward(self, output, target):
        # output = output['flow_preds']
        # target = target['flows'][:, 0]
        lossvalue = 0

        if type(output) is tuple or type(output) is list:
            #target = self.div_flow * target
            for i, flow_pred_ in enumerate(output):
                target_ = self.multiScales[i](target)
                lossvalue += self.loss_weights[i]*self.loss(flow_pred_, target_)
        else:
            lossvalue += self.loss(output, target)
        return lossvalue


@FUNCTIONAL_REGISTRY.register()
class MultiScaleLossV2(nn.Module):
    """
    Multi-scale loss for optical flow estimation.
    Used in **DICL** (https://papers.nips.cc/paper/2020/hash/add5aebfcb33a2206b6497d53bc4f309-Abstract.html)

    Parameters
    ----------
    norm : str
        The norm to use for the loss. Can be either "l2", "l1" or "robust"
    weights : list
        The weights to use for each scale
    extra_mask : torch.Tensor
        A mask to apply to the loss. Useful for removing the loss on the background
    use_valid_range : bool
        Whether to use the valid range of flow values for the loss
    valid_range : list
        The valid range of flow values for each scale
    """

    @configurable
    def __init__(
        self,
        norm="l1",
        weights=(1, 0.5, 0.25),
        average="mean",
        resize_flow="upsample",
        extra_mask=None,
        use_valid_range=True,
        valid_range=None,
    ):
        super(MultiScaleLossV2, self).__init__()

        self.norm = norm.lower()
        assert self.norm in ("l1", "l2", "robust"), "Norm must be one of L1, L2, Robust"

        self.weights = weights
        self.extra_mask = extra_mask
        self.use_valid_range = use_valid_range
        self.valid_range = valid_range
        self.average = average
        self.resize_flow = resize_flow

    @classmethod
    def from_config(cls, cfg):
        return {
            "norm": cfg.NORM,
            "weights": cfg.WEIGHTS,
            "average": cfg.AVERAGE,
            "resize_flow": cfg.RESIZE_FLOW,
            "extra_mask": cfg.EXTRA_MASK,
            "use_valid_range": cfg.USE_VALID_RANGE,
            "valid_range": cfg.VALID_RANGE,
        }

    def forward(self, preds, label):

        if isinstance(preds, dict):
            #flow2, flow3, flow4, flow5, flow6
            pred = [preds['level6'],preds['level5'],preds['level4'],preds['level4'],preds['level2']]

        if label.shape[1] == 3:
            """Ignore valid mask for Multiscale Loss."""
            label = label[:, :2, :, :]

        loss = 0
        b, c, h, w = label.size()

        if (
            (type(pred) is not tuple)
            and (type(pred) is not list)
            and (type(pred) is not set)
        ):
            pred = {pred}

        for i, level_pred in enumerate(pred):

            #print(f"pred: {level_pred.shape} target: {label.shape}")

            if self.resize_flow.lower() == "upsample":
                real_flow = F.interpolate(
                    level_pred, (h, w), mode="bilinear", align_corners=True
                )
                real_flow[:, 0, :, :] = real_flow[:, 0, :, :] * (w / level_pred.shape[3])
                real_flow[:, 1, :, :] = real_flow[:, 1, :, :] * (h / level_pred.shape[2])
                target = label
            
            elif self.resize_flow.lower() == "downsample":
                # down sample ground truth following irr solution
                # https://github.com/visinf/irr/blob/master/losses.py#L16
                b,c,h,w = level_pred.shape

                target = F.adaptive_avg_pool2d(label, [h, w])
                real_flow = level_pred


            if self.norm == "l2":
                loss_value = torch.norm(real_flow - target, p=2, dim=1)

            elif self.norm == "robust":
                loss_value = (real_flow - target).abs().sum(dim=1) + 1e-8
                loss_value = loss_value**0.4

            elif self.norm == "l1":
                loss_value = (real_flow - target).abs().sum(dim=1)

            if self.use_valid_range and self.valid_range is not None:

                with torch.no_grad():
                    mask = (target[:, 0, :, :].abs() <= self.valid_range[i][1]) & (
                        target[:, 1, :, :].abs() <= self.valid_range[i][0]
                    )
            else:
                with torch.no_grad():
                    mask = torch.ones(target[:, 0, :, :].shape).type_as(target)

            #print(f"pred: {real_flow.shape} target: {target.shape} loss map: {loss_value.shape} weight: {self.weights[i]}")

            loss_value = loss_value * mask.float()

            if self.extra_mask is not None:
                val = self.extra_mask > 0
                loss_value = loss_value[val]
                
            if self.average.lower() ==  "mean":  
                level_loss = loss_value.mean() * self.weights[i]

            elif self.average.lower() ==  "sum":
                level_loss = loss_value.sum() / b * self.weights[i]

            loss += level_loss
            #print(40*'-')

        #print(f"loss: {loss} total preds: {len(pred)}")
        loss = loss / len(pred)

        return loss