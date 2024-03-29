## Exploring Training Recipes and Transformer Neural Networks for Optical Flow Estimation

[[official thesis report]](https://www.proquest.com/openview/9a6963d105d9c4fb27e20c8a49d02a0f/1?pq-origsite=gscholar&cbl=18750&diss=y) | [[unpublished thesis draft]](https://tinyurl.com/prajnan-ms-thesis-draft)

This thesis makes the following contributions:
- An empirical study of pre-training, dataset scheduling, and data augmentations on _**four generations of optical flow models**_ to provide an _**Improved Training Recipe**_. 
- Understanding the efficacy of Transformer Neural Networks for the optical flow estimation task.

The majority of the code is supported by the [EzFlow](https://github.com/neu-vi/ezflow) PyTorch Library which was developed as a prerequisite for the thesis study. This repository contains the training configuration files for all the experiments and the implementation of **NAT-GM** and **SCCFlow** [_end-to-end transformer_](https://github.com/prajnan93/optical-flow-msthesis/tree/main/nnflow/models) architectures for optical flow estimation.

____

The _**improved training recipe**_ can be found here: [kubric_improved_aug](https://github.com/neu-vi/ezflow/blob/main/configs/trainers/_base_/kubric_improved_aug.yaml)
____

### Four Generations of Optical Flow Models

<p align="center">
    <br>
    <img src="./assets/flow_models.jpg"/>
    <br>
</p>
____

### Getting Started

- Follow instructions to setup EzFlow and the conda environment from [EzFlow Getting Started](https://github.com/neu-vi/ezflow/blob/main/CONTRIBUTING.rst#get-started)
- Install the following additional packages:
  ```
  pip install git+https://github.com/huggingface/transformers
  pip3 install natten -f https://shi-labs.com/natten/wheels/cu113/torch1.10.1/index.html 
  pip install timm
  ```
- If `natten` package fails to install, follow the setup directions from: https://www.shi-labs.com/natten/ 
____

The pretrained checkpoints for the improved results will be published in the [EzFlow](https://github.com/neu-vi/ezflow) repository.

____

### References

- [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852)
- [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://arxiv.org/abs/1709.02371)
- [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039)
- [GMFlow: Learning Optical Flow via Global Matching](https://arxiv.org/abs/2111.13680)
- [Disentangling Architecture and Training for Optical Flow](https://arxiv.org/abs/2203.10712)
- [ViT: Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Dino ViT: Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814)
- [Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143)
- [Dilated Neighborhood Attention Transformer](https://arxiv.org/abs/2209.15001)
- [Kubric](https://github.com/google-research/kubric/tree/main/challenges/optical_flow)

____

### Citation

```bibtex


@article{
    author={Goswami,Prajnan},
    year={2022},
    title={Exploring Training Recipes and Transformer Neural Networks for Optical Flow Estimation},
    journal={ProQuest Dissertations and Theses},
    url={https://www.proquest.com/docview/2789009042?pq-origsite=gscholar&fromopenview=true},
}

@software{Shah_EzFlow_A_modular_2021,
    author = {Shah, Neelay and Goswami, Prajnan and Jiang, Huaizu},
    license = {MIT},
    month = {11},
    title = {{EzFlow: A modular PyTorch library for optical flow estimation using neural networks}},
    url = {https://github.com/neu-vig/ezflow},
    year = {2021}
}
```
