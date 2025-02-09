import time
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from poker_cnn_3class import PokerCNN3Class

class PokerImageClassifier3Class:
    def __init__(self, model_path='best_poker_cnn_3class.pth', num_classes=3, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model = self.load_model(model_path, num_classes, self.device)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path, num_classes, device):
        model = PokerCNN3Class(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0)  # 添加批次维度
        return image

    def infer_image(self, image_path):
        image = self.preprocess_image(image_path)
        image = image.to(self.device)

        with torch.no_grad():
            start_time = time.time()
            output = self.model(image)
            end_time = time.time()

            # 获取预测类别和置信度
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)
            confidence = probabilities[0][predicted].item()

            elapsed_time = end_time - start_time
            print(f"Inference for {image_path} took {elapsed_time:.4f} seconds")

            return predicted.item(), confidence

    def detect_image(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)

            # 获取预测类别和置信度
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)
            confidence = probabilities[0][predicted].item()

            return predicted.item(), confidence

    def infer_images(self, image_dir):
        image_dir = Path(image_dir)
        images = list(image_dir.glob('*.jpg'))
        num_images = len(images)

        if num_images == 0:
            print("No images found in the directory.")
            return

        total_time = 0.0

        with torch.no_grad():
            for img_path in images:
                image = self.preprocess_image(img_path)
                image = image.to(self.device)

                start_time = time.time()
                output = self.model(image)
                end_time = time.time()

                total_time += (end_time - start_time)

                # 获取预测类别和置信度
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(probabilities, 1)
                confidence = probabilities[0][predicted].item()

                print(f"Image: {img_path.name}, Predicted Class: {predicted.item()}, Confidence: {confidence:.4f}")

        avg_time = total_time / num_images
        print(f"Average Inference Time per Image: {avg_time:.4f} seconds")
        print(f"Total Inference Time: {total_time:.4f} seconds")

if __name__ == "__main__":
    # 创建分类器实例
    classifier = PokerImageClassifier3Class()

    # 图片目录
    image_dir = 'datasets/region1/images'

    # 进行推理并计算识别速度
    classifier.infer_images(image_dir)

    # 示例：推理单张图片
    single_image_path = 'datasets/region1/images/108_0000000001.jpg'
    predicted_class, confidence = classifier.infer_image(single_image_path)
    print(f"Single Image: {single_image_path}, Predicted Class: {predicted_class}, Confidence: {confidence:.4f}")
