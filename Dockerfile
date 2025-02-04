FROM ultralytics/ultralytics:8.2.98

# Atualiza os repositórios e instala o Python, pip, git e outras dependências básicas
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Cria um link simbólico para facilitar o uso do comando "python"
RUN ln -s /usr/bin/python3 /usr/bin/python

# Define o diretório de trabalho
WORKDIR /app

# Define argumentos de build para as credenciais do Kaggle
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

# Cria o diretório .kaggle e gera o arquivo kaggle.json com as credenciais
RUN mkdir -p /root/.kaggle && \
    echo '{"username":"'$KAGGLE_USERNAME'","key":"'$KAGGLE_KEY'"}' > /root/.kaggle/kaggle.json && \
    chmod 600 /root/.kaggle/kaggle.json



# Copia o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instala as dependências listadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mantém o container em execução
CMD ["tail", "-f", "/dev/null"]
