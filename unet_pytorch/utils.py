import matplotlib.pyplot as plt
import numpy as np
import torch

def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed set to {seed}.")


def print_model_info(model=None):
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

def plot_overlay_4x2(batch, alpha=0.5):
    images, masks = batch
    fig, ax = plt.subplots(4, 2, figsize=(16, 16))
    for i in range(4):
        for j in range(2):
            image = images[i * 2 + j]
            mask = masks[i * 2 + j]
            ax[i, j].imshow(image[0], cmap='gray')
            ax[i, j].imshow(mask[0], cmap='jet', alpha=alpha)
            ax[i, j].axis('off')
    plt.show()

def plot_predictions(x, y, y_pred, num_examples_to_plot=3, shuffle=True):
    fig, ax = plt.subplots(num_examples_to_plot, 3, figsize=(9, num_examples_to_plot * 3))
    for i in range(num_examples_to_plot):
        if shuffle:
            idx = np.random.randint(0, len(x))
        else:
            idx = i
        image = x[idx]
        mask = y[idx]
        pred_mask = y_pred[idx]
        ax[i][0].imshow(image[0].squeeze(), cmap="gray")
        ax[i][1].imshow(image[0].squeeze(), cmap="gray")
        ax[i][1].imshow(mask.squeeze(), cmap="jet", alpha=0.5)
        ax[i][2].imshow(image[0].squeeze(), cmap="gray")
        ax[i][2].imshow(pred_mask.squeeze(), cmap="jet", alpha=0.5)
        ax[i][0].axis("off")
        ax[i][0].set_title("Image")
        ax[i][1].axis("off")
        ax[i][1].set_title("Ground Truth")
        ax[i][2].axis("off")
        ax[i][2].set_title("Prediction")
    plt.tight_layout()
    plt.show()

def plot_one_example(x, y):
    y = y.type(torch.int64)
    y_one_hot = torch.nn.functional.one_hot(y, 3).transpose(1, 3).squeeze(-1)
    # now i need to rotate this image 90 degrees clockwise
    y_one_hot = y_one_hot.transpose(2, 3)

    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    ax[0].imshow(x[0].squeeze(), cmap="gray")
    ax[1].imshow(y_one_hot[0][0].squeeze(), cmap="gray")
    ax[2].imshow(y_one_hot[0][1].squeeze(), cmap="gray")
    ax[3].imshow(y_one_hot[0][2].squeeze(), cmap="gray")
    ax[4].imshow(x[0].squeeze(), cmap="gray")
    ax[4].imshow(y.squeeze(), cmap="jet", alpha=0.5)
    

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[3].axis("off")
    ax[4].axis("off")

    # add titles
    ax[0].set_title("Image")
    ax[1].set_title("Background")
    ax[2].set_title("Prostate inner (PZ)")
    ax[3].set_title("Prostate outer (TZ)")
    ax[4].set_title("Overlay")

    plt.tight_layout()
    plt.show()
