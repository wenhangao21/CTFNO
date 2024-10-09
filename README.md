# Coordinate Transform Fourier Neural Operators for Symmetries in Physical Modelings (CTFNO)

This repository contains the code implementation for the paper titled "Coordinate Transform Fourier Neural Operators for Symmetries in Physical Modelings", **W.Gao**, R.Xu, H.Wang, Y.Liu. TMLR, 2024.

Full paper on [OpenReview](https://openreview.net/forum?id=pMD7A77k3i).

<p align="center">
  <img src="https://wenhangao21.github.io/images/TMLR_CTNO.png" alt="Figure" width="400"/>
</p>

**Bibtex**:
```bibtex
@article{gao2024coordinate,
  title={Coordinate Transform Fourier Neural Operators for Symmetries in Physical Modelings},
  author={Wenhan Gao and Ruichen Xu and Hong Wang and Yi Liu},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

## Introduction 
The laws of physics, including partial differential equations (PDEs), remain unchanged regardless of the coordinate system employed to depict them, and symmetries sometimes can be natural to illuminate in other coordinate systems.

By applying simple coordinate transformations, we can more effectively capture and exploit the inherent symmetries in models like the Fourier Neural Operator (FNO) and Convolutional Neural Networks (CNN).


## How to Run
Below are commands for training the Darcy flow equation:
```python
python train.py --which_example=darcy --which_model={FNO, PTFNO, GFNO, RFNO} --random_seed=0
```
- FNO: Fourier Neural Operator from Zongyi Li et al.

- GFNO: Group-FNO from Jacob Helwig et al.

- PTFNO: Coordinate Transform FNO

- RFNO: Radial FNO (the kernel is isotropic, so naturally equivariant to rotations)

## Structure

- `train.py`: The main script for training the model.
  
- `data/`: Contains the data for training

- `models/`: Contains all the models. All the models assume to operate on 2D rectangular domains. Input output shape (B, C, S_x, S_y): Batchsize, Channels, spatial resolution, spatial resolution.

- `settings/`: Contains the set up (hyperparameters, basic properties of the data etc..) in `properties.py`; sets up the dataloaders in `data_module.py`; sets up the models in `model_module.py`

- `utils/`: Contains auxiliary functions such as calculating loss

- `TrainedModels/`: This folder will be created by the program to store training properties, record losses, and save the final trained weights.

## Preparing Data

Please refer to the FNO paper and their repository to obtain the Darcy flow data. Place the .mat files under `data/` and change the file names in `data_module.py` as necessary. The heat equation data, along with the code, will be released soon.