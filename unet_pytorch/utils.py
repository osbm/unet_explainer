import matplotlib.pyplot as plt

def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed set to {seed}.")
def get_parameter_number(model=None):
    # pytorch_total_params
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()

    # calculate megabytes that this model would occupy in memory
    total_megabytes = total * 4 / 1024 / 1024
    print(f"Total number of trainable parameters: {total:,} ({total_megabytes:.2f}MB)")

def plot_image_batch(image, mask, num_examples_to_plot=3):
    fig, ax = plt.subplots(2, num_examples_to_plot, figsize=(num_examples_to_plot * 3, 6))
    for i in range(num_examples_to_plot):
        ax[0][i].imshow(image[i, 0], cmap="gray")
        ax[1][i].imshow(mask[i, 0], cmap="gray")
        ax[0][i].axis("off")
        ax[1][i].axis("off")
    plt.tight_layout()
    plt.show()

def plot_history(history: dict):
    keys = list(history.keys())
    plot_types = [key.replace("train_", "") for key in keys if "train" in key]
    
    for plot_type in plot_types:
        plt.figure(figsize=(10, 5))
        plt.plot(history[f"train_{plot_type}"], label=f"train_{plot_type}")
        plt.plot(history[f"valid_{plot_type}"], label=f"valid_{plot_type}")
        plt.legend()
        plt.title(f"{plot_type.capitalize()} History")
        plt.xlabel("Epochs")
        plt.ylabel(plot_type)
        plt.show()


def plot_overlay(image, mask, alpha=0.5):
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.show()

def plot_overlay_4x4(batch, alpha=0.5):
    images, masks = batch
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        for j in range(4):
            image = images[i * 4 + j]
            mask = masks[i * 4 + j]
            ax[i, j].imshow(image[0], cmap='gray')
            ax[i, j].imshow(mask[0], cmap='jet', alpha=alpha)
            ax[i, j].axis('off')
    plt.show()
