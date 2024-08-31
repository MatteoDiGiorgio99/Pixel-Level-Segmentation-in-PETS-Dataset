# PETS Dataset Segmentation using UNet

**Author:** Matteo Di Giorgio - 353719

This repository contains an implementation of a segmentation model aimed at performing pixel-level segmentation on the Oxford-IIIT Pet Dataset, which contains images of pets along with their corresponding segmentation masks.

## Table of Contents

- [Project Objective](#project-objective)

- [Dataset](#dataset)

- [Installation](#installation)

- [Project Structure](#project-structure)

- [Training](#training)

- [Studies](#studies)

- [Run with HPC](#run-with-hpc)

## Project Objective

The goal of this project is to develop a deep learning model capable of performing pixel-level segmentation on the PETS dataset. The main objectives are:

- Segmentation of pets from RGB images.

- The network should output a segmentation mask for each pet.

- Only a single pet is present in each image.

### Importing project

**Clone the repository:**

```bash

git clone https://github.com/MatteoDiGiorgio99/Pixel-Level-Segmentation-in-PETS-Dataset

```

## Dataset

The dataset used in this project is the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which provides segmentation masks for a large variety of pets. Ensure you download and prepare the dataset correctly before training the model.

## Installation

### Prerequisites

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

Make sure you have the following installed on your system:

- Python 3.8 or higher

- pip (Python package installer)

### Dependencies

Install the necessary dependencies using `pip`:

```bash

pip  install  -r  requirements.txt

```

_Note:_ Ensure you have installed necessary libraries like `torch`, `numpy`...
If necessary, update versions and dependencies.

## Project Structure

### Project

- `Config.py`: Contains configuration parameters for the training.

- `Dataset.py`: Handles loading and preprocessing of the PETS dataset.

- `mainUnet.py`: The main entry point for training and evaluating the segmentation model.

- `MetricLoss.py`: Implements custom loss functions and metrics such as Intersection over Union (IoU) and L1 distance.

- `Model.py`: Defines the architecture of the UNet model.

- `Train.py`: Manages the training loop and model optimization.

- `Utils.py`: Includes utility functions used across the project.

### Notebooks

- `Evaluate.ipynb`: Used to display training and model performance.

- `ResultsPaper.ipynb`: Notebook containing the studies carried out and their results.

## Training

To train the segmentation model:

1. Configure your parameters in `Config.py`.
2. Run the training script --> run `mainUnet.py`
3. Evaluate the results with `Evaluate.ipynb`

Training logs and model checkpoints will be saved in the specified directory.

## Studies

To view the results obtained from the various model studies, please refer to `ResultsPaper.ipynb`
Graphs can be viewed via the **TensorBoard** function in the notebook

In the `runs` folder, all studies carried out with:

- Version of the code related to the study (.py)

- Output obtained (.O)

- Script for job submission (.sh)

- History Training (.pickle)

- Training Process Images

**You can make use of the already trained models of the various versions of the code via the following [OneDrive](https://univpr-my.sharepoint.com/:f:/g/personal/matteo_digiorgio_studenti_unipr_it/EueF_oNikpBMtJ_GdgwCFBQBjD6_5OOfCmVzY04BWcnSLQ?e=gRdeHm) link**

## Run with HPC

You can use the corresponding scripts for execution with HPC

Example:

1. **Login HPC** (Password access is allowed only within the University network (160.78.0.0/16). Outside this context it is necessary to use the University [VPN](https://wiki.asi.unipr.it/dokuwiki/doku.php?id=guide_utente:internet:guida_vpn "https://wiki.asi.unipr.it/dokuwiki/doku.php?id=guide_utente:internet:guida_vpn").

```bash
ssh name.surname@login.hpc.unipr.it
```

2. **Upload the project** to your profile in a folder of your choice

3. **Go to Folder**

```bash
cd folder
```

4. **Job Submission**

```bash
sbatch mainUnet.sh
```
