## Lung Image Diagnosis Using AI
This project uses machine learning (specifically a Convolutional Neural Network) to analyze X-ray images of the lungs to predict whether a patient has pneumonia or not. The model is trained on a dataset of medical images and uses deep learning techniques to classify them.
This project aims to assist in diagnosing lung conditions using AI by leveraging Convolutional Neural Networks (CNNs) for image classification. The model has been trained to detect pneumonia in X-ray images based on features learned during training.

##Features
Convolutional Neural Network (CNN): The core architecture of the model, used to extract hierarchical features from images.
Image Preprocessing: Includes resizing, normalization, and data augmentation techniques to improve model performance.
Model Training: The model is trained on a dataset of lung X-ray images to classify pneumonia.
Model File: The trained model (model.h5) is ready for use in predictions.

##Setup Instructions
1. Clone the Repository
git clone https://github.com/Aboodx0191/LungImageDiagnosis.git
2. Install Required Libraries
Ensure you have Python 3.x installed. Then, use the following commands to set up your environment and install the required dependencies:

pip install -r requirements.txt

## Dataset
To train the model, you’ll need a dataset of lung X-ray images. You can download the dataset from the following sources:

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Training the Model
Once you have the dataset and the required libraries installed, you can train the model using the model.py file. The model will be trained on your local machine.
The model will be saved as model.h5 once training is complete.which will be provided bellow

## Model File
The model file `model.h5` can be downloaded from [Google Drive Link](https://drive.google.com/file/d/1D17QuVUatBDpa97Hyso1acJgd5ZSbOk1/view?usp=drive_link)
