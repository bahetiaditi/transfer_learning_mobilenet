# utils/visualization.py
import matplotlib.pyplot as plt
import torch

import matplotlib.pyplot as plt

def visualize_prediction(image, ground_truth, prediction, epoch, num_epochs):

    if image.dim() == 4:
        image = image[0]
    if ground_truth.dim() == 4 or ground_truth.dim() == 3:
        ground_truth = ground_truth[0]
    if prediction.dim() == 4 or prediction.dim() == 3:
        prediction = prediction[0]

    image = image.cpu()
    ground_truth = ground_truth.cpu()
    prediction = prediction.cpu()


    ground_truth = ground_truth.squeeze()
    prediction = prediction.squeeze()
    prediction = prediction > 0.5


    if image.dim() == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.numpy())
    axs[0].set_title('Input Image')
    axs[1].imshow(ground_truth.numpy(), cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(prediction.numpy(), cmap='gray')
    axs[2].set_title('Model Prediction')
    plt.suptitle(f'Epoch {epoch+1}/{num_epochs}')
    plt.show()

import matplotlib.pyplot as plt

def visualize_test_prediction(image, ground_truth, prediction):

    if image.dim() == 4:
        image = image[0]
    if ground_truth.dim() == 4 or ground_truth.dim() == 3:
        ground_truth = ground_truth[0]
    if prediction.dim() == 4 or prediction.dim() == 3:
        prediction = prediction[0]


    image = image.cpu()
    ground_truth = ground_truth.cpu()
    prediction = prediction.cpu()


    ground_truth = ground_truth.squeeze()
    prediction = prediction.squeeze()
    prediction= prediction > 0.5


    if image.dim() == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image.numpy())
    axs[0].set_title('Input Image')
    axs[1].imshow(ground_truth.numpy(), cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(prediction.numpy(), cmap='gray')
    axs[2].set_title('Model Prediction')
    plt.suptitle(f'Test Prediction')
    plt.show()

