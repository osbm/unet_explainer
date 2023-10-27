import torch
from tqdm import tqdm
from monai.metrics import DiceHelper
from torchmetrics import JaccardIndex


def fit_model(
    model=None,
    train_loader=None,
    valid_loader=None,
    optimizer=None,
    loss=None,
    device=None,
    epochs=10
):
    
    best_valid_loss = float("inf")
    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_dice": [],
        "valid_dice": [],
        "train_iou": [],
        "valid_iou": [],
    }

    dice_metric = DiceHelper(include_background=True, softmax=True, reduction="mean")
    iou_metric = JaccardIndex(num_classes=3, task="multiclass").to(device)
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = 0
        train_dice_scores = 0
        train_iou_scores = 0
        for images, masks in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_value = loss(outputs, masks)
            loss_value.backward()

            dice_score, _ = dice_metric(outputs, masks)

            iou_score = iou_metric(outputs.argmax(dim=1).squeeze(), masks.squeeze())
            train_dice_scores += dice_score.item()
            train_iou_scores += iou_score.item()
            train_epoch_loss += loss_value.item()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()


        train_epoch_loss /= len(train_loader)
        train_dice_scores /= len(train_loader)
        train_iou_scores /= len(train_loader)

        history["train_loss"].append(train_epoch_loss)
        history["train_dice"].append(train_dice_scores)
        history["train_iou"].append(train_iou_scores)

        model.eval()
        valid_epoch_loss = 0
        valid_dice_scores = 0
        valid_iou_scores = 0

        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Valid Epoch {epoch+1}/{epochs}", leave=False):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss_value = loss(outputs, masks)

                dice_score, _ = dice_metric(outputs, masks)
                iou_score = iou_metric(outputs.argmax(dim=1), masks.squeeze())


                valid_dice_scores += dice_score.item()
                valid_iou_scores += iou_score.item()
                valid_epoch_loss += loss_value.item()

        valid_dice_scores /= len(valid_loader)
        valid_iou_scores /= len(valid_loader)
        valid_epoch_loss /= len(valid_loader)

        history["valid_dice"].append(valid_dice_scores)
        history["valid_iou"].append(valid_iou_scores)
        history["valid_loss"].append(valid_epoch_loss)


        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            print("Saving better model to best_model.pth")
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss {train_epoch_loss:.4f}/{valid_epoch_loss:.4f}, "
            f"dice {train_dice_scores:.4f}/{valid_dice_scores:.4f}, "
            f"iou {train_iou_scores:.4f}/{valid_iou_scores:.4f}"
        )

    return model, history


def predict(model, test_loader=None, device=None, final_activation="softmax", calculate_scores=True):
    model.eval()

    x = []
    y_pred = []
    y = []
    
    if calculate_scores:
        dice_metric = DiceHelper(include_background=True, softmax=True, reduction="mean")
        iou_metric = JaccardIndex(num_classes=3, task="multiclass").to(device)

        test_dice_score = 0
        test_iou_score = 0
    test_dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images, masks = images.to(device), masks.to(device)
            

            outputs = model(images)
            
            if calculate_scores:
                dice_score, _ = dice_metric(outputs, masks)
                iou_score = iou_metric(outputs.argmax(dim=1), masks.squeeze())

            if final_activation == "softmax":
                outputs = torch.softmax(outputs, dim=1)
            elif final_activation == "sigmoid":
                outputs = torch.sigmoid(outputs)
            elif final_activation == "none":
                pass
            else:
                raise ValueError("final_activation must be either softmax, sigmoid or none")

            test_dice_score += dice_score.item()
            test_iou_score += iou_score.item()
            test_dice_scores.append(dice_score.item())
            y_pred.append(outputs.argmax(dim=1).detach().cpu().unsqueeze(1))
            y.append(masks.detach().cpu())
            x.append(images.detach().cpu())
    

    # now sort x, y and y_pred by dice score
    if calculate_scores:
        test_dice_scores = torch.tensor(test_dice_scores)
        _, indices = torch.sort(test_dice_scores, descending=True)
        x = torch.cat(x)[indices]
        y = torch.cat(y)[indices]
        y_pred = torch.cat(y_pred)[indices]
    else:
        x = torch.cat(x)
        y = torch.cat(y)
        y_pred = torch.cat(y_pred)

    if calculate_scores:
        test_dice_score /= len(test_loader)
        test_iou_score /= len(test_loader)
        print(f"Dice score: {test_dice_score}")
        print(f"IoU score: {test_iou_score}")

    return x, y, y_pred
