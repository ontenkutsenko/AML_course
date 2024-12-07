
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torch
# from tqdm.notebook import tqdm

def train(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torchvision.models,
    num_epochs: int,
    device: str,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    lr: float,
    path_to_save_model: str,
    num_classes: int

) -> None:

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    #Define trainanble layers
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)
    optimizer = optimizer(model.parameters(), lr=lr)

    global_acc_train = []
    global_loss_train = []
    global_acc_val = []
    global_loss_val = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        counter = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            counter += 1

            if counter % 50 == 0:
                print(f"Train: Epoch [{epoch+1}/{num_epochs}], \nStep [{counter}/{len(train_loader)}], \nLoss: {loss.item():.4f}, n\Accuracy: {(100 * correct / total):.4f}")

        global_acc_train.append(100 * correct / total)
        global_loss_train.append(running_loss / len(train_loader))
        # validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        global_acc_val.append(100 * correct / total)
        global_loss_val.append(running_loss / len(val_loader))
        print(f"Val: Epoch [{epoch+1}/{num_epochs}], n\Loss: {(running_loss / len(val_loader)):.4f}, n\Accuracy: {(100 * correct / total):.4f}")

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(global_acc_train, label='Train Accuracy')
        ax[0].plot(global_acc_val, label='Validation Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        ax[1].plot(global_loss_train, label='Train Loss')
        ax[1].plot(global_loss_val, label='Validation Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        plt.show()


    #save model weights
    torch.save(model.state_dict(), path_to_save_model)

def test_eval(
    test_loader,
    model,
    device: str,
    path_to_save_model: str
):
    model.load_state_dict(torch.load(path_to_save_model))
    #test eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {(100 * correct / total):.4f}")