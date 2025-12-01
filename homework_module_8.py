import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torch.utils.data import Dataset
from PIL import Image

# ============================================
# REPRODUCIBILITY SETUP
# ============================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HairDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                self.image_paths.append(os.path.join(label_dir, img_name))
                self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================
# QUESTION 1: Model Architecture and Loss Function
# ============================================
print("\n" + "="*60)
print("QUESTION 1: Which loss function to use?")
print("="*60)
print("Answer: nn.BCEWithLogitsLoss()")
print("Reason: For binary classification with single output neuron,")
print("        BCEWithLogitsLoss combines sigmoid activation and BCE loss")
print("        for numerical stability.")


# ============================================
# MODEL DEFINITION
# ============================================
class HairTypeCNN(nn.Module):
    def __init__(self):
        super(HairTypeCNN, self).__init__()

        # Convolutional layer: 32 filters, kernel size (3,3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()

        # Max pooling layer: pooling size (2,2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After conv (3x3) on 200x200: output = 198x198
        # After maxpool (2x2): output = 99x99
        # So flattened size = 32 * 99 * 99 = 313,632
        self.flatten_size = 32 * 99 * 99

        # Fully connected layer: 64 neurons with relu
        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.relu2 = nn.ReLU()

        # Output layer: 1 neuron (NO activation - BCEWithLogitsLoss will apply sigmoid)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input shape: (batch_size, 3, 200, 200)
        x = self.conv1(x)      # -> (batch_size, 32, 198, 198)
        x = self.relu1(x)
        x = self.pool(x)       # -> (batch_size, 32, 99, 99)

        # Flatten
        x = x.view(-1, self.flatten_size)  # -> (batch_size, 313632)

        x = self.fc1(x)        # -> (batch_size, 64)
        x = self.relu2(x)
        x = self.fc2(x)        # -> (batch_size, 1) - raw logits

        return x


# Initialize model
model = HairTypeCNN().to(device)


# ============================================
# QUESTION 2: Total number of parameters
# ============================================
print("\n" + "="*60)
print("QUESTION 2: Total number of parameters")
print("="*60)

from torchsummary import summary
summary(model, input_size=(3, 200, 200))

# Count parameters manually
conv1_params = 3 * 32 * 3 * 3 + 32  # weights + bias
fc1_params = 313632 * 64 + 64       # weights + bias
fc2_params = 64 * 1 + 1             # weights + bias

print(f"Conv1 parameters: {conv1_params:,}")
print(f"FC1 parameters: {fc1_params:,}")
print(f"FC2 parameters: {fc2_params:,}")
print(f"Total (manual): {conv1_params + fc1_params + fc2_params:,}")

# Count using PyTorch
total_params = sum(p.numel() for p in model.parameters())
print(f"Total (PyTorch): {total_params:,}")
print(f"\nAnswer: {total_params} ≈ 20,073,473")


# ============================================
# DATA PREPARATION - Initial (No Augmentation)
# ============================================
print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

train_transforms = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )  # ImageNet normalization
])

train_dataset = HairDataset(
    data_dir='data/train',
    transform=train_transforms
)

validation_dataset = HairDataset(
    data_dir='data/test',
    transform=train_transforms
)

# Load datasets
#train_dataset = datasets.ImageFolder('data/train', transform=train_transforms)
#validation_dataset = datasets.ImageFolder('data/test', transform=train_transforms)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")
print(f"Classes: {train_dataset.classes}")

# Create data loaders
batch_size = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


# ============================================
# TRAINING SETUP
# ============================================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)


# ============================================
# TRAINING LOOP (First 10 epochs - No Augmentation)
# ============================================
print("\n" + "="*60)
print("TRAINING - First 10 epochs (No Augmentation)")
print("="*60)

num_epochs = 10
history = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)  # Ensure labels are float and have shape (batch_size, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        # For binary classification with BCEWithLogitsLoss, apply sigmoid to outputs before thresholding for accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train / total_train
    history['loss'].append(epoch_loss)
    history['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history['val_loss'].append(val_epoch_loss)
    history['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")


# ============================================
# QUESTION 3: Median of training accuracy
# ============================================
print("\n" + "="*60)
print("QUESTION 3: Median of training accuracy")
print("="*60)
print(f"Training accuracies: {history['acc']}")
median_acc = np.median(history['acc'])
print(f"Median training accuracy: {median_acc:.4f}")
print(f"Answer: {median_acc:.2f}")


# ============================================
# QUESTION 4: Standard deviation of training loss
# ============================================
print("\n" + "="*60)
print("QUESTION 4: Standard deviation of training loss")
print("="*60)
print(f"Training losses: {history['loss']}")
std_loss = np.std(history['loss'])
print(f"Standard deviation of training loss: {std_loss:.4f}")
print(f"Answer: {std_loss:.3f}")


# ============================================
# DATA AUGMENTATION
# ============================================
print("\n" + "="*60)
print("DATA AUGMENTATION - Training for 10 more epochs")
print("="*60)

# Create augmented training transforms
train_transforms_aug = transforms.Compose([
    transforms.RandomRotation(50),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Reload training dataset with augmentation
#train_dataset_aug = datasets.ImageFolder('data/train', transform=train_transforms_aug)

train_dataset_aug = HairDataset(
    data_dir='data/train',
    transform=train_transforms_aug
)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)

# Continue training for 10 more epochs
print("Continuing training with augmentation...")
history_aug = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader_aug:
        images, labels = images.to(device), labels.to(device)
        labels = labels.float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataset_aug)
    epoch_acc = correct_train / total_train
    history_aug['loss'].append(epoch_loss)
    history_aug['acc'].append(epoch_acc)

    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = correct_val / total_val
    history_aug['val_loss'].append(val_epoch_loss)
    history_aug['val_acc'].append(val_epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
          f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")


# ============================================
# QUESTION 5: Mean of test loss with augmentation
# ============================================
print("\n" + "="*60)
print("QUESTION 5: Mean of test loss (validation loss) with augmentation")
print("="*60)
print(f"Test losses (with augmentation): {history_aug['val_loss']}")
mean_test_loss = np.mean(history_aug['val_loss'])
print(f"Mean test loss: {mean_test_loss:.4f}")
print(f"Answer: {mean_test_loss:.2f}")


# ============================================
# QUESTION 6: Average test accuracy for last 5 epochs
# ============================================
print("\n" + "="*60)
print("QUESTION 6: Average test accuracy for last 5 epochs (6-10) with augmentation")
print("="*60)
last_5_epochs = history_aug['val_acc'][5:10]  # epochs 6-10 (indices 5-9)
print(f"Test accuracies for epochs 6-10: {last_5_epochs}")
avg_test_acc = np.mean(last_5_epochs)
print(f"Average test accuracy (last 5 epochs): {avg_test_acc:.4f}")
print(f"Answer: {avg_test_acc:.2f}")


# ============================================
# SUMMARY OF ANSWERS
# ============================================
print("\n" + "="*60)
print("SUMMARY OF ANSWERS")
print("="*60)
print(f"Question 1: nn.BCEWithLogitsLoss()")
print(f"Question 2: {total_params:,} parameters")
print(f"Question 3: Median training accuracy = {median_acc:.2f}")
print(f"Question 4: Std of training loss = {std_loss:.3f}")
print(f"Question 5: Mean test loss (augmentation) = {mean_test_loss:.2f}")
print(f"Question 6: Avg test acc (epochs 6-10, augmentation) = {avg_test_acc:.2f}")
print("="*60)
