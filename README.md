**Marine-Animal-Sound-Detection using CNN and ResNet50**

## Overview
This project aims to develop a marine animal sound detection system using Convolutional Neural Networks (CNN) and the ResNet50 architecture. The system is designed to identify various marine animal sounds from audio recordings, contributing to marine conservation efforts by monitoring and studying marine ecosystems.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- Librosa
- NumPy
- Pandas
- Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/Marine-Animal-sound-detection.git
   cd Marine-Animal-sound-detection
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (if applicable) and place it in the `data` directory.

## Usage
1. Preprocess the data:
   ```bash
   python preprocess_data.py
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

3. Evaluate the model:
   ```bash
   python evaluate_model.py
   ```

4. Make predictions:
   ```bash
   python predict.py <path_to_audio_file>
   ```

## Dataset
The dataset consists of labeled audio recordings of marine animal sounds. It may be obtained from [https://www.watkins-marine.com/], or a custom dataset can be used. Ensure that the dataset is structured properly and follows the required format. We used a custom dataset of 72 species.

## Model
The model architecture used for this project is ResNet50, a deep convolutional neural network known for its performance on image classification tasks. The audio spectrograms are treated as images and fed into the network for classification.

## Results
We obtained a accuracy of 97% on Training Dataset and 92% on Validation Dataset.

## License
This project is licensed under the [MIT License](LICENSE).
