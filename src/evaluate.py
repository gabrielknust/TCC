import torch
import torchvision.models as models
import torch.nn as nn
from thop import profile
import time
import pandas as pd

# --- SUA CLASSE DE MODELO (Copie exatamente igual ao treino) ---
class MobileNetV2WithEarlyExits(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2WithEarlyExits, self).__init__()
        self.mobilenet_base = models.mobilenet_v2(weights=None)
        
        # Ajuste 32x32
        self.mobilenet_base.features[0][0] = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.part1 = self.mobilenet_base.features[:7]
        self.part2 = self.mobilenet_base.features[7:14]
        self.part3 = self.mobilenet_base.features[14:]

        self.early_exit_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(32, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
        
        self.early_exit_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(96, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
        
        self.final_classifier = nn.Linear(self.mobilenet_base.last_channel, num_classes)
        self.mobilenet_base.classifier = nn.Identity()

    # Não precisamos do forward completo aqui, vamos medir por partes
    def forward(self, x):
        pass

def format_flops(flops):
    """Converte FLOPs para M (Mega) ou G (Giga)"""
    if flops > 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    return f"{flops/1e6:.2f} MFLOPs"

def measure_computational_cost():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de teste: {device}")
    
    model = MobileNetV2WithEarlyExits().to(device)
    model.eval()

    # Imagem dummy 32x32 (Batch size 1)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    print("\n" + "="*60)
    print("1. MEDINDO COMPLEXIDADE TEÓRICA (FLOPs e Parâmetros)")
    print("="*60)

    # --- CÁLCULO DOS BLOCOS INDIVIDUAIS ---
    # Nota: profile retorna (MACs, Params). FLOPs ~= 2 * MACs. 
    # Mas a convenção é reportar MACs como "FLOPs" em muitos papers de MobileNet.
    
    # Backbone Part 1
    flops_p1, params_p1 = profile(model.part1, inputs=(dummy_input, ), verbose=False)
    # Output da Part 1 (precisamos saber o shape para passar para a próxima)
    out_p1 = model.part1(dummy_input)
    
    # Exit Head 1
    flops_h1, params_h1 = profile(model.early_exit_1, inputs=(out_p1, ), verbose=False)

    # Backbone Part 2
    flops_p2, params_p2 = profile(model.part2, inputs=(out_p1, ), verbose=False)
    out_p2 = model.part2(out_p1)

    # Exit Head 2
    flops_h2, params_h2 = profile(model.early_exit_2, inputs=(out_p2, ), verbose=False)

    # Backbone Part 3
    flops_p3, params_p3 = profile(model.part3, inputs=(out_p2, ), verbose=False)
    out_p3 = model.part3(out_p2)
    
    # Preparação para final classifier (pool + flatten)
    # Como são operações simples, as vezes o thop ignora, mas vamos ignorar o custo do pool aqui
    out_p3_flat = torch.flatten(nn.functional.adaptive_avg_pool2d(out_p3, (1, 1)), 1)
    
    # Final Head
    flops_hf, params_hf = profile(model.final_classifier, inputs=(out_p3_flat, ), verbose=False)

    # --- CÁLCULO CUMULATIVO (O que realmente importa) ---
    total_flops_exit1 = flops_p1 + flops_h1
    total_flops_exit2 = flops_p1 + flops_p2 + flops_h2
    total_flops_final = flops_p1 + flops_p2 + flops_p3 + flops_hf

    total_params_exit1 = params_p1 + params_h1
    # Nota: Para params, o Exit 1 não usa os parâmetros do Exit 2, mas no 'disco' o modelo tem tudo.
    # Aqui calculamos "quantos parâmetros foram usados para a inferência".
    total_params_exit2 = params_p1 + params_p2 + params_h2
    total_params_final = params_p1 + params_p2 + params_p3 + params_hf

    print(f"{'SAÍDA':<15} | {'FLOPs (MACs)':<15} | {'PARAMS USADOS':<15} | {'ECONOMIA (FLOPs)'}")
    print("-" * 65)
    print(f"{'Exit 1':<15} | {format_flops(total_flops_exit1):<15} | {int(total_params_exit1):<15} | {100*(1 - total_flops_exit1/total_flops_final):.1f}%")
    print(f"{'Exit 2':<15} | {format_flops(total_flops_exit2):<15} | {int(total_params_exit2):<15} | {100*(1 - total_flops_exit2/total_flops_final):.1f}%")
    print(f"{'Final':<15} | {format_flops(total_flops_final):<15} | {int(total_params_final):<15} | 0.0%")


    print("\n" + "="*60)
    print("2. MEDINDO LATÊNCIA REAL (Tempo de Inferência)")
    print(f"Executando 1000 repetições para média estável...")
    print("="*60)

    # Warmup (Aquecer a GPU/CPU)
    for _ in range(100):
        _ = model.part1(dummy_input)

    # Medição Exit 1
    start = time.time()
    for _ in range(1000):
        f1 = model.part1(dummy_input)
        _ = model.early_exit_1(f1)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.time()
    latency_e1 = (end - start) / 1000

    # Medição Exit 2
    start = time.time()
    for _ in range(1000):
        f1 = model.part1(dummy_input)
        f2 = model.part2(f1)
        _ = model.early_exit_2(f2)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.time()
    latency_e2 = (end - start) / 1000

    # Medição Final
    start = time.time()
    for _ in range(1000):
        f1 = model.part1(dummy_input)
        f2 = model.part2(f1)
        f3 = model.part3(f2)
        f3 = nn.functional.adaptive_avg_pool2d(f3, (1, 1))
        f3 = torch.flatten(f3, 1)
        _ = model.final_classifier(f3)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.time()
    latency_final = (end - start) / 1000

    print(f"{'SAÍDA':<15} | {'LATÊNCIA (ms)':<15} | {'FPS (img/s)':<15} | {'SPEEDUP'}")
    print("-" * 65)
    print(f"{'Exit 1':<15} | {latency_e1*1000:.3f} ms       | {1/latency_e1:.1f}           | {latency_final/latency_e1:.2f}x mais rápido")
    print(f"{'Exit 2':<15} | {latency_e2*1000:.3f} ms       | {1/latency_e2:.1f}           | {latency_final/latency_e2:.2f}x mais rápido")
    print(f"{'Final':<15} | {latency_final*1000:.3f} ms       | {1/latency_final:.1f}           | 1.00x")

if __name__ == "__main__":
    measure_computational_cost()