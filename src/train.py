import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import random_split
import os
import logging
from datetime import datetime

def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

class MobileNetV2WithEarlyExits(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2WithEarlyExits, self).__init__()
        self.mobilenet_base = models.mobilenet_v2(weights="IMAGENET1K_V1")

        self.part1 = self.mobilenet_base.features[:7]
        self.part2 = self.mobilenet_base.features[7:14]
        self.part3 = self.mobilenet_base.features[14:]

        self.early_exit_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.early_exit_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.final_classifier = nn.Linear(self.mobilenet_base.last_channel, num_classes)
        self.mobilenet_base.classifier = nn.Identity()

    def forward(self, x):
        features1 = self.part1(x)
        output1 = self.early_exit_1(features1)

        features2 = self.part2(features1)
        output2 = self.early_exit_2(features2)

        final_features = self.part3(features2)
        final_features = nn.functional.adaptive_avg_pool2d(final_features, (1, 1))
        final_features = torch.flatten(final_features, 1)
        final_output = self.final_classifier(final_features)

        return output1, output2, final_output

if __name__ == "__main__":
    setup_logging()

    logging.info("="*50)
    logging.info("INICIANDO NOVA EXECUÇÃO DE TREINAMENTO")
    logging.info("="*50)

    num_epochs = 10
    batch_size = 64
    lr = 0.0001
    
    logging.info(f"Hiperparâmetros: Épocas={num_epochs}, Batch Size={batch_size}, Learning Rate={lr}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    logging.info("Carregando e dividindo o dataset CIFAR-10...")
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    train_dataset, val_dataset = random_split(full_trainset, [train_size, val_size])
    logging.info(f"Dataset dividido em: {train_size} para treino, {val_size} para validação.")

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    logging.info(f"Dataset de teste carregado com {len(testset)} imagens.")

    mobilenet = MobileNetV2WithEarlyExits(num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"Modelo Utilizado: {mobilenet.__class__.__name__}")
    logging.info(f"Device Utilizado: {device}")

    mobilenet = mobilenet.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mobilenet.parameters(), lr=lr)
    
    logging.info("Iniciando loop de treinamento...")
    for epoch in range(num_epochs):
        mobilenet.train()
        running_loss = 0.0
        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            early_outputs_1, early_outputs_2, final_outputs = mobilenet(images)
            loss_1 = criterion(early_outputs_1, labels)
            loss_2 = criterion(early_outputs_2, labels)
            loss_final = criterion(final_outputs, labels)
            loss = 0.2 * loss_1 + 0.3 * loss_2 + 0.5 * loss_final

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_dataset)
        logging.info(f"Epoch {epoch+1} finalizada. Loss de treino: {epoch_loss:.4f}")

        mobilenet.eval()
        val_loss = 0.0
        correct_1, correct_2, correct_final, total = 0, 0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                early_outputs_1, early_outputs_2, final_outputs = mobilenet(images)
                
                loss_1 = criterion(early_outputs_1, labels)
                loss_2 = criterion(early_outputs_2, labels)
                loss_final = criterion(final_outputs, labels)
                loss = 0.2 * loss_1 + 0.3 * loss_2 + 0.5 * loss_final
                val_loss += loss.item() * images.size(0)

                _, pred_1 = torch.max(early_outputs_1, 1)
                correct_1 += (pred_1 == labels).sum().item()
                _, pred_2 = torch.max(early_outputs_2, 1)
                correct_2 += (pred_2 == labels).sum().item()
                _, pred_final = torch.max(final_outputs, 1)
                correct_final += (pred_final == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_dataset)
        val_acc_1 = 100 * correct_1 / total
        val_acc_2 = 100 * correct_2 / total
        val_acc_final = 100 * correct_final / total
        
        logging.info(f"VAL => Loss: {val_loss:.4f} | Acc Early 1: {val_acc_1:.2f}% | Acc Early 2: {val_acc_2:.2f}% | Acc Final: {val_acc_final:.2f}%")

    logging.info("Treinamento finalizado.")
    logging.info("Avaliando o modelo final no conjunto de teste...")
    
    mobilenet.eval()
    test_loss = 0.0
    correct_1, correct_2, correct_final, total = 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            early_outputs_1, early_outputs_2, final_outputs = mobilenet(images)
            
            loss_1 = criterion(early_outputs_1, labels)
            loss_2 = criterion(early_outputs_2, labels)
            loss_final = criterion(final_outputs, labels)
            loss = 0.2 * loss_1 + 0.3 * loss_2 + 0.5 * loss_final
            test_loss += loss.item() * images.size(0)

            _, pred_1 = torch.max(early_outputs_1, 1)
            correct_1 += (pred_1 == labels).sum().item()
            _, pred_2 = torch.max(early_outputs_2, 1)
            correct_2 += (pred_2 == labels).sum().item()
            _, pred_final = torch.max(final_outputs, 1)
            correct_final += (pred_final == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(testset)
    test_acc_1 = 100 * correct_1 / total
    test_acc_2 = 100 * correct_2 / total
    test_acc_final = 100 * correct_final / total
    
    logging.info("-" * 50)
    logging.info("Resultados Finais no Conjunto de Teste:")
    logging.info(f"TEST => Loss: {test_loss:.4f} | Acc Early 1: {test_acc_1:.2f}% | Acc Early 2: {test_acc_2:.2f}% | Acc Final: {test_acc_final:.2f}%")
    logging.info("-" * 50)

    SAVE_DIR = 'saved_model'
    os.makedirs(SAVE_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'mobilenet_final.pth')
    torch.save(mobilenet.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Modelo final salvo com sucesso em: {MODEL_SAVE_PATH}")
    logging.info("Execução finalizada.")