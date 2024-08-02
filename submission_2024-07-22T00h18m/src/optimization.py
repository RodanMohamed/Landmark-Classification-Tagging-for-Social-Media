import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification)
    """
    loss = nn.CrossEntropyLoss()
    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        opt = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "adam":
        opt = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "adamw":
        opt = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt

def optimize(data_loaders, model, optimizer, loss_fn, n_epochs, save_path, interactive_tracking=True):
    """
    Optimize the model.
    
    :param data_loaders: dictionary containing 'train', 'valid', and 'test' DataLoaders
    :param model: the model to be trained
    :param optimizer: optimizer instance
    :param loss_fn: loss function
    :param n_epochs: number of epochs to train
    :param save_path: path to save the best model
    :param interactive_tracking: whether to print training progress
    """
    best_val_loss = float('inf')

    # Use a StepLR scheduler to reduce the learning rate by a factor of 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0

        for inputs, labels in tqdm(data_loaders['train'], desc=f"Epoch {epoch+1}/{n_epochs}", leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        scheduler.step()

        train_loss = running_loss / len(data_loaders['train'].dataset)
        train_acc = running_corrects.double() / total_train

        model.eval()
        val_loss = 0.0
        running_corrects = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in data_loaders['valid']:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_loss = val_loss / len(data_loaders['valid'].dataset)
        val_acc = running_corrects.double() / total_val

        if interactive_tracking:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{n_epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, LR: {current_lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

######################################################################################
#                                     TESTS
######################################################################################
import pytest

@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)

def test_get_loss():
    loss = get_loss()
    assert isinstance(loss, nn.CrossEntropyLoss), f"Expected cross entropy loss, found {type(loss)}"

def test_get_optimizer_type(fake_model):
    opt = get_optimizer(fake_model)
    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"

def test_get_optimizer_is_linked_with_model(fake_model):
    opt = get_optimizer(fake_model)
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])

def test_get_optimizer_returns_adam(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam")
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam optimizer, got {type(opt)}"

def test_get_optimizer_sets_learning_rate(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)
    assert opt.param_groups[0]["lr"] == 0.123, "get_optimizer is not setting the learning rate appropriately. Check your code."

def test_get_optimizer_sets_momentum(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)
    assert opt.param_groups[0]["momentum"] == 0.123, "get_optimizer is not setting the momentum appropriately. Check your code."

def test_get_optimizer_sets_weight_decay(fake_model):
    opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)
    assert opt.param_groups[0]["weight_decay"] == 0.123, "get_optimizer is not setting the weight_decay appropriately. Check your code."
