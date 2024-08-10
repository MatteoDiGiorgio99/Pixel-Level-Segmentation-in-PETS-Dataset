
# PETS Dataset Segmentation using UNet

This repository contains an implementation of a segmentation model aimed at performing pixel-level segmentation on the PETS dataset, which contains images of pets along with their corresponding segmentation masks.

## Table of Contents

- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Project Objective

The goal of this project is to develop a deep learning model capable of performing pixel-level segmentation on the PETS dataset. The main objectives are:
- Segmentation of pets from RGB images.
- The network should output a segmentation mask for each pet.
- Only a single pet is present in each image.

## Dataset

The dataset used in this project is the [PETS dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), which provides segmentation masks for a large variety of pets. Ensure you download and prepare the dataset correctly before training the model.

## Installation

### Prerequisites

Make sure you have the following installed on your system:
- Python 3.8 or higher
- pip (Python package installer)

### Dependencies

Install the necessary dependencies using `pip`:

```bash
pip install -r requirements.txt
```

*Note:* If `requirements.txt` is not provided, ensure you have installed necessary libraries like `torch`, `numpy`, and others commonly used in deep learning projects.

## Usage

### Running the Segmentation Model

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/unet-pets-segmentation.git
   cd unet-pets-segmentation
   ```

2. **Prepare your dataset:**

   Download the PETS dataset and organize it as per the expected format. Refer to `Dataset.py` for details on how the data should be structured.

3. **Train the model:**

   To start training the segmentation model, run:

   ```bash
   python mainUnet.py --train --config path_to_config_file
   ```

4. **Evaluate the model:**

   After training, you can evaluate the model by running:

   ```bash
   python mainUnet.py --evaluate --model path_to_trained_model
   ```

## Project Structure

- `Config.py`: Contains configuration parameters for the model, training, and evaluation.
- `Dataset.py`: Handles loading and preprocessing of the PETS dataset.
- `mainUnet.py`: The main entry point for training and evaluating the segmentation model.
- `MetricLoss.py`: Implements custom loss functions and metrics such as Intersection over Union (IoU) and L1 distance.
- `Model.py`: Defines the architecture of the UNet model. You can experiment with different architectures here.
- `Train.py`: Manages the training loop and model optimization.
- `Utils.py`: Includes utility functions used across the project.

## Training

To train the segmentation model:

1. Configure your parameters in `Config.py`.
2. Ensure your data is correctly preprocessed and structured as per the PETS dataset requirements.
3. Run the training script as mentioned in the usage section.

Training logs and model checkpoints will be saved in the specified directory.

## Evaluation

To evaluate the performance of the trained model, use the evaluation script provided. The model's performance will be assessed using metrics such as Intersection over Union (IoU) and L1 distance.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
