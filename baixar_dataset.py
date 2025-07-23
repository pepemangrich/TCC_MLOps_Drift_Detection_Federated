from datasets import load_dataset

# Isso baixa o CIFAR-10 e salva em cache padrão (Hugging Face)
print("Baixando CIFAR-10 via Hugging Face...")
load_dataset("cifar10")
print("Download completo.")