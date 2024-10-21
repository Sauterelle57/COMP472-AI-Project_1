import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.modules.container import Sequential
import torch.nn as nn

def get_dataset(train_size: int = 500, test_size: int = 100) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionner à 224x224
        transforms.ToTensor(),  # Convertir en tenseur PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
    ])

    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_data = Subset(train_data, range(train_size * 10))
    test_data = Subset(test_data, range(test_size * 10))

    train_dataloader = DataLoader(train_data, batch_size=train_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_size)

    return train_dataloader, test_dataloader

def init_resnet18() -> Sequential:
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    for param in resnet.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = resnet.to(device)

    return resnet

def extract_features(data: DataLoader, model: Sequential) -> torch.Tensor:
    model.eval()
    features = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for i, (images, _) in enumerate(data):
            images = images.to(device)
            output = model(images)
            features.append(output.squeeze())
            print(f"Processed batch {i+1}")
    return torch.cat(features)

def resnet_18(train_loader: DataLoader, test_loader: DataLoader):
    resnet = init_resnet18()

    print("\nExtracting features in train set...")
    train_features = extract_features(train_loader, resnet)
    print("\nExtracting features in test set...")
    test_features = extract_features(test_loader, resnet)

    return train_features, test_features
