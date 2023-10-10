import matplotlib.pyplot as plt



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