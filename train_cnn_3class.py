import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from poker_cnn_3class import PokerCNN3Class


class PokerDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        self.cached_images = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        full_img_path = self.img_dir / img_path

        # 检查缓存
        if img_path in self.cached_images:
            image = self.cached_images[img_path]
        else:
            # 读取图片
            try:
                image = Image.open(full_img_path).convert('RGB')
                self.cached_images[img_path] = image  # 缓存图片
            except Exception as e:
                print(f"Failed to open image {full_img_path}: {e}")
                return None, None

        if self.transform:
            image = self.transform(image)

        # 解析标签
        try:
            label_str = img_path.split('_')[0]
            label = int(label_str)
            if label not in [0, 1, 2]:
                raise ValueError(f"Invalid label: {label} for image {img_path}")
        except (IndexError, ValueError) as e:
            print(f"Error parsing label for image {img_path}: {e}")
            return None, None

        return image, label


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda'):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            if inputs is None or labels is None:
                continue  # 跳过无效的数据

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                if inputs is None or labels is None:
                    continue  # 跳过无效的数据

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_poker_cnn_3class.pth')

        print()


def main():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = PokerDataset(
        img_dir='datasets/train_3class',  # 修改为新的数据集路径
        transform=transform
    )

    val_dataset = PokerDataset(
        img_dir='datasets/val_3class',  # 修改为新的数据集路径
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 创建模型
    model = PokerCNN3Class(num_classes=3).to(device)

    # 加载预训练模型权重
    # model.load_state_dict(torch.load('best_poker_cnn_3class.pth', map_location=device))
    print("Pre-trained model loaded successfully.")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=40,
        device=device
    )


if __name__ == '__main__':
    main()
