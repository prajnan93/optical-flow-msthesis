from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ezflow.functional import FUNCTIONAL_REGISTRY

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
                          flow_div: float = 1.0,
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

            loss_map += loss_function(i_pred, cur_target, **kwargs) * cur_valid

            if reduction == 'mean':
                loss += loss_map.sum() / (cur_valid.sum() + 1e-8) * cur_weight
            elif reduction == 'sum':
                loss += loss_map.sum() / b * cur_weight

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
                 flow_div: float = 1.,
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
        preds_dict = {
            'level2': preds[0],
            'level3': preds[1],
            'level4': preds[2],
            'level5': preds[3],
            'level6': preds[4]
        }

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
        self.div_flow = 0.05
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
            target = self.div_flow * target
            for i, flow_pred_ in enumerate(output):
                target_ = self.multiScales[i](target)
                lossvalue += self.loss_weights[i]*self.loss(flow_pred_, target_)
        else:
            lossvalue += self.loss(output, target)
        return lossvalue