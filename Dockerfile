FROM tensorflow/tensorflow:2.15.0-gpu

# Evita interação ao instalar pacotes e configura timezone automaticamente
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Dublin

# Atualiza os repositórios e instala Python, pip, git e dependências adicionais
RUN apt-get update && apt-get install -y \
    git \
    screen \
    nano \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho
WORKDIR /app

# Define argumentos de build para credenciais Kaggle
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

# Configura Kaggle API
RUN mkdir -p /root/.kaggle && \
    echo '{"username":"'$KAGGLE_USERNAME'","key":"'$KAGGLE_KEY'"}' > /root/.kaggle/kaggle.json && \
    chmod 600 /root/.kaggle/kaggle.json

# Copia o requirements.txt e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mantém o container rodando
CMD ["tail", "-f", "/dev/null"]
