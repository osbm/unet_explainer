import torch
from tqdm import tqdm

def fit_model(
    model=None,
    train_loader=None,
    valid_loader=None,
    optimizer=None,
    loss=None,
    device=None,
    epochs=10
):
    model.train()
    best_valid_loss = float("inf")
    history = {
        "train_loss": [],
        "valid_loss": [],
    }
    for epoch in range(epochs):
        train_epoch_loss = 0

        for images, masks in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_value = loss(outputs, masks)
            loss_value.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            train_epoch_loss += loss_value.item()

        train_epoch_loss /= len(train_loader)

        history["train_loss"].append(train_epoch_loss)

        model.eval()
        valid_epoch_loss = 0

        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Valid Epoch {epoch+1}/{epochs}", leave=False):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss_value = loss(outputs, masks)

                valid_epoch_loss += loss_value.item()


        valid_epoch_loss /= len(valid_loader)
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            print("Saving better model to best_model.pth")
            torch.save(model.state_dict(), "best_model.pth")
        history["valid_loss"].append(valid_epoch_loss)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss {train_epoch_loss:.4f}/{valid_epoch_loss:.4f}, "
        )

    return model, history


def predict(model, test_loader=None, device=None, final_activation=None):
    model.eval()

    x = []
    y_pred = []
    y = []
    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images, masks = images.to(device), masks.to(device)
            

            outputs = model(images)
            if final_activation is not None:
                outputs = final_activation(outputs)
            y_pred.append(outputs.argmax(dim=1).detach().cpu())
            y.append(masks.detach().cpu())
            x.append(images.detach().cpu())

    x = torch.cat(x)
    y = torch.cat(y)
    y_pred = torch.cat(y_pred)

    return x, y, y_pred
