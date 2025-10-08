import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

CIFAR10_CLASSES = ('avião', 'automóvel', 'pássaro', 'gato', 'cervo', 'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão')

class MobileNetV2WithEarlyExits(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2WithEarlyExits, self).__init__()
        self.mobilenet_base = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.part1 = self.mobilenet_base.features[:7]
        self.part2 = self.mobilenet_base.features[7:14]
        self.part3 = self.mobilenet_base.features[14:]
        self.early_exit_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(32, 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes)
        )
        self.early_exit_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(96, 128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes)
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

def predict_images(image_paths):
    MODEL_SAVE_PATH = os.path.join('saved_model', 'mobilenet_final.pth')

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Erro: Arquivo do modelo não encontrado em '{MODEL_SAVE_PATH}'")
        return

    print(f"Carregando modelo de: {MODEL_SAVE_PATH}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MobileNetV2WithEarlyExits(num_classes=10)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("\n--- Iniciando Predições ---")
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output1, output2, final_output = model(image_tensor)

            _, pred1_idx = torch.max(output1, 1)
            _, pred2_idx = torch.max(output2, 1)
            _, final_pred_idx = torch.max(final_output, 1)

            pred1_class = CIFAR10_CLASSES[pred1_idx.item()]
            pred2_class = CIFAR10_CLASSES[pred2_idx.item()]
            final_pred_class = CIFAR10_CLASSES[final_pred_idx.item()]

            print(f"\nArquivo: {image_path}")
            print(f"  - Predição Saída 1: {pred1_class}")
            print(f"  - Predição Saída 2: {pred2_class}")
            print(f"  - Predição Saída Final: {final_pred_class}")

        except FileNotFoundError:
            print(f"\nErro: Arquivo não encontrado em '{image_path}'")
        except Exception as e:
            print(f"\nErro ao processar o arquivo '{image_path}': {e}")
            
    print("\n--- Predições finalizadas ---")

if __name__ == '__main__':
    pasta_de_imagens = 'images'
    if not os.path.isdir(pasta_de_imagens):
        print(f"Erro: O diretório '{pasta_de_imagens}' não foi encontrado.")
        print("Por favor, crie a pasta e adicione imagens ou corrija o caminho.")
    else:
        imagens_para_testar = []
        extensoes_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        
        print(f"Procurando por imagens na pasta: '{pasta_de_imagens}'...")
        
        for nome_arquivo in os.listdir(pasta_de_imagens):
            if nome_arquivo.lower().endswith(extensoes_validas):
                caminho_completo = os.path.join(pasta_de_imagens, nome_arquivo)
                imagens_para_testar.append(caminho_completo)
        
        if not imagens_para_testar:
            print("Nenhuma imagem encontrada na pasta especificada.")
        else:
            print(f"{len(imagens_para_testar)} imagem(ns) encontrada(s).")
            predict_images(imagens_para_testar)