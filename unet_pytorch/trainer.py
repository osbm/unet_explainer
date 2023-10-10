from typing import Tuple
from torch import nn
import torch
import monai
from tqdm import tqdm

def fit_model(
    model: nn.Module=None,
    train_loader: torch.utils.data.DataLoader=None,
    valid_loader: torch.utils.data.DataLoader=None,
    optimizer: torch.optim.Optimizer=None,
    loss: nn.Module=None,
    device: torch.types.Device=None,
    epochs: int=10
) -> Tuple[nn.Module, dict]:
    model.train()

    history = {
        # loss
        "train_loss": [],
        "valid_loss": [],
        # accuracy
        "train_accuracy": [],
        "valid_accuracy": [],
        # dice score
        # "train_dice_score": [],
        # "valid_dice_score": [],
        # jacard score
        # "train_jaccard_score": [],
        # "valid_jaccard_score": [],
    }
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        # epoch_dice_score = 0
        # epoch_jaccard_score = 0

        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)

            loss_value = loss(outputs, masks)
            loss_value.backward()

            optimizer.step()

            accuracy = (outputs.argmax(dim=1) == masks).float().mean()
            # dice_score = monai.metrics.compute_meandice(
            #     y_pred=outputs.argmax(dim=1),
            #     y=mask,
            #     include_background=False,
            # )
            # jaccard_score = monai.metrics.compute_meandice(
            #     y_pred=outputs.argmax(dim=1),
            #     y=mask,
            #     include_background=False,
            # )

            epoch_loss += loss_value.item()
            epoch_accuracy += accuracy.item()
            # epoch_dice_score += dice_score.item()
            # epoch_jaccard_score += jaccard_score.item()

        epoch_loss /= len(train_loader)
        epoch_accuracy /= len(train_loader)
        # epoch_dice_score /= len(train_loader)
        # epoch_jaccard_score /= len(train_loader)

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)
        # history["train_dice_score"].append(epoch_dice_score)
        # history["train_jaccard_score"].append(epoch_jaccard_score)

        print(
            f"train epoch {epoch + 1}: "
            f"loss {epoch_loss:.4f}, "
            f"accuracy {epoch_accuracy:.4f}, "
            # f"dice score {epoch_dice_score:.4f}, "
            # f"jaccard score {epoch_jaccard_score:.4f}, "
        )

        model.eval()

        epoch_loss = 0
        epoch_accuracy = 0
        # epoch_dice_score = 0
        # epoch_jaccard_score = 0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Valid Epoch {epoch+1}/{epochs}", leave=False):
                images, masks = batch
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)

                loss_value = loss(outputs, masks)

                accuracy = (outputs.argmax(dim=1) == masks).float().mean()
                # dice_score = monai.metrics.compute_meandice(
                #     y_pred=outputs.argmax(dim=1),
                #     y=mask,
                #     include_background=False,
                # )
                # jaccard_score = monai.metrics.compute_meandice(
                #     y_pred=outputs.argmax(dim=1),
                #     y=mask,
                #     include_background=False,
                # )

                epoch_loss += loss_value.item()
                epoch_accuracy += accuracy.item()
                # epoch_dice_score += dice_score.item()
                # epoch_jaccard_score += jaccard_score.item()

            epoch_loss /= len(valid_loader)
            epoch_accuracy /= len(valid_loader)
            # epoch_dice_score /= len(valid_loader)
            # epoch_jaccard_score /= len(valid_loader)

            history["valid_loss"].append(epoch_loss)
            history["valid_accuracy"].append(epoch_accuracy)
            # history["valid_dice_score"].append(epoch_dice_score)
            # history["valid_jaccard_score"].append(epoch_jaccard_score)

            print(
                f"valid epoch {epoch + 1}: "
                f"loss {epoch_loss:.4f}, "
                f"accuracy {epoch_accuracy:.4f}, "
                # f"dice score {epoch_dice_score:.4f}, "
                # f"jaccard score {epoch_jaccard_score:.4f}, "
            )

    return model, history


def predict(
    model: nn.Module=None,
    test_loader: torch.utils.data.DataLoader=None,
    device: torch.types.Device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    model.eval()

    y_pred = []
    y = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)

            y_pred.append(outputs.argmax(dim=1).detach().cpu())
            y.append(masks.detach().cpu())

    y_pred = torch.cat(y_pred)
    y = torch.cat(y)

    return y_pred, y