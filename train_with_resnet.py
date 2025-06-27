import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torchvision_models import resnet18  # Replace with actual filename if not in same file
import numpy as np
from PIL import Image

def generate_dummy_images():
    os.makedirs("dataset/real", exist_ok=True)
    os.makedirs("dataset/fake", exist_ok=True)
    for i in range(5):
        real_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        fake_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        Image.fromarray(real_img).save(f"dataset/real/real_{i}.jpg")
        Image.fromarray(fake_img).save(f"dataset/fake/fake_{i}.jpg")
    print("Dummy images generated.")

generate_dummy_images()
# Dataset path
DATASET_DIR = "dataset"
MODEL_SAVE_PATH = "models/deepfake_cnn_model.pth"
NUM_CLASSES = 2

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# DataLoader
dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Load pre-trained ResNet18 and modify final layer
model = resnet18(num_classes=1000, pretrained='imagenet')
model.last_linear = nn.Linear(model.last_linear.in_features, NUM_CLASSES)  # 2 outputs: real, fake
model = model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train
for epoch in range(5):
    total_loss = 0
    model.train()
    for inputs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved at: {MODEL_SAVE_PATH}")
