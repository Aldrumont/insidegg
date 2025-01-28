import os
import kaggle
import shutil


# Nome do dataset no Kaggle
dataset_name = "aldrumont/insidegg-v2"

download_path = "videos"

# Baixar o dataset e descompactar
print(f"Baixando dataset {dataset_name}...")
kaggle.api.dataset_download_files(dataset_name, path='.', unzip=True)

# Mover o arquivo .xlsx para a raiz
for file in os.listdir(download_path):
    if file.endswith(".xlsx"):
        shutil.move(os.path.join(download_path, file), file)
        print(f"Arquivo {file} movido para a raiz do diretório.")

print("Download concluído! Os arquivos de vídeo estão na pasta 'videos' e o arquivo .xlsx está na raiz.")
