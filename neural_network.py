import os
import random
import pickle
import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, TimeDistributed
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# ============================
# Configurações Iniciais
# ============================
seed_number = 2024
random.seed(seed_number)
np.random.seed(seed_number)
tf.random.set_seed(seed_number)

# Parâmetros e caminhos
FRAMES_FOLDER_PATH = 'frames'
EGG_DATA_SHEET_PATH = 'dados_ovos.xlsx'
FINAL_FRAME_SHAPE = (50, 50)
GRAYSCALE = False
POSSIBLE_OBJECTIVE_COLUMNS = ['Massa Total Aferido', 'Clara', 'Gema', 'Casca', 'Gema + Clara']
OBJECTIVE_COLUMNS = ['Massa Total Aferido', 'Clara', 'Gema + Clara']
NUM_ORIGINALS_PER_EGG = 30
MAX_FRAMES_BY_COLOR = 8
COLORS = ['blue', 'green', 'red', 'white']

# ============================
# Funções Utilitárias
# ============================
def save_data_pickle(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def load_data_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def send_telegram_message(message):
    bot_token = '797390018:AAHWSVCdBvXnERxnz4kI4ACvrfu_Rq7xLxc'
    chat_id = '626444855'
    response = requests.get(f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={message}')
    if response.status_code == 200:
        print('Mensagem enviada com sucesso!')
    else:
        print(f'Falha ao enviar a mensagem. Código de status: {response.status_code}')

def send_telegram_image(image_path):
    bot_token = '797390018:AAHWSVCdBvXnERxnz4kI4ACvrfu_Rq7xLxc'
    chat_id = '626444855'
    url = f'https://api.telegram.org/bot{bot_token}/sendPhoto'
    with open(image_path, 'rb') as image_file:
        response = requests.post(url, data={'chat_id': chat_id}, files={'photo': image_file})
    if response.status_code == 200:
        print('Imagem enviada com sucesso!')
    else:
        print(f'Falha ao enviar a imagem. Código de status: {response.status_code}')

def send_telegram_file(file_path, caption=""):
    bot_token = '797390018:AAHWSVCdBvXnERxnz4kI4ACvrfu_Rq7xLxc'
    chat_id = '626444855'
    url = f'https://api.telegram.org/bot{bot_token}/sendDocument'
    with open(file_path, 'rb') as file:
        response = requests.post(url, data={'chat_id': chat_id, 'caption': caption}, files={'document': file})
    if response.status_code == 200:
        print('Arquivo enviado com sucesso!')
    else:
        print(f'Falha ao enviar o arquivo. Código de status: {response.status_code}, Resposta: {response.text}')

# ============================
# Funções de Pré-processamento e Augmentação
# ============================
def augment_image(image, dx, angulo, fator_brilho, h_flip, v_flip, rotate, pct_noise,
                  apply_blur=False, shear_range=False, swap_channels=False):
    altura, largura = image.shape[:2]
    # Adição de ruído
    if pct_noise:
        num_pixels = int(altura * largura * pct_noise)
        y_coords = np.random.randint(0, altura, num_pixels)
        x_coords = np.random.randint(0, largura, num_pixels)
        if len(image.shape) == 3:
            noise = np.random.randint(0, 256, (num_pixels, image.shape[2]), dtype=np.uint8)
        else:
            noise = np.random.randint(0, 256, num_pixels, dtype=np.uint8)
        image[y_coords, x_coords] = noise

    # Translação
    M_shift = np.float32([[1, 0, dx], [0, 1, 0]])
    image = cv2.warpAffine(image, M_shift, (largura, altura))

    # Flips
    if h_flip:
        image = cv2.flip(image, 1)
    if v_flip:
        image = cv2.flip(image, 0)

    # Rotação (usando rotações em múltiplos de 90)
    if rotate == 90:
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    elif rotate == 180:
        image = cv2.flip(image, -1)
    elif rotate == 270:
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)

    # Ajuste de brilho
    image = cv2.convertScaleAbs(image, alpha=fator_brilho)

    # Shear
    if shear_range:
        shear_value = np.random.uniform(-shear_range, shear_range)
        M = np.array([[1, shear_value, 0], [0, 1, 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M, (largura, altura))

    # Blur
    if apply_blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Swap de canais somente se a imagem tiver 3 canais (não para grayscale)
    if swap_channels and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Garante que se estivermos em grayscale, a imagem tenha shape (altura, largura, 1)
    if GRAYSCALE and len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    return image

def augment_image_individually(X, Y, max_shift=8, max_rotation=1, brightness_range=0.5, num_augmentations=10, pct_noise=0.02):
    X_data, Y_data = [], []
    for original_frames, y in tqdm(zip(X, Y), total=len(X), desc="Augmenting samples"):
        X_data.append(original_frames)
        Y_data.append(y)
        for _ in range(num_augmentations):
            augmented_frames = []
            for frame in original_frames:
                dx = np.random.randint(-max_shift, max_shift)
                angulo = np.random.uniform(-max_rotation, max_rotation)
                fator_brilho = np.random.uniform(1 - brightness_range, 1 + brightness_range)
                h_flip = np.random.choice([True, False])
                v_flip = np.random.choice([True, False])
                rotate = np.random.choice([0, 90, 180, 270])
                noise_factor = random.uniform(0, 1)
                pct_noise_sample = noise_factor * pct_noise
                apply_blur = np.random.choice([True, False])
                shear_range = 0
                swap_channels = np.random.choice([True, False])
                augmented_frame = augment_image(frame, dx, angulo, fator_brilho, h_flip, v_flip,
                                                rotate, pct_noise_sample, apply_blur, shear_range, swap_channels)
                augmented_frames.append(augmented_frame)
            X_data.append(np.array(augmented_frames))
            Y_data.append(y)
    return np.array(X_data), np.array(Y_data)

def select_frames_with_spacing(total_frames=36, num_selected_frames=8):
    start_index = random.randint(0, total_frames - 1)
    spacing = total_frames // num_selected_frames
    selected_indices = [(start_index + i * spacing) % total_frames for i in range(num_selected_frames)]
    return selected_indices

def load_frames(egg_index, colors=['green', 'red', 'blue', 'white'], videos=[1, 2], base_path=FRAMES_FOLDER_PATH,
                resize_ratio=1.0, grayscale=False, max_frames_by_color=6, final_frame_shape=(100, 100), write_index=False):
    frames = []
    if not isinstance(videos, list):
        videos = [videos]
    # Obter caminhos dos vídeos
    videos_paths = []
    for i in videos:
        videos_folder = os.listdir(os.path.join(base_path, f'egg_{egg_index}'))
        videos_paths.append(videos_folder[i - 1])
    for video_path in videos_paths:
        for color in colors:
            path = os.path.join(base_path, f'egg_{egg_index}', video_path, color)
            frame_files = sorted(os.listdir(path))
            if len(frame_files) < max_frames_by_color:
                raise ValueError(f"Número insuficiente de frames na cor {color} para o ovo {egg_index}.")
            selected_indices = select_frames_with_spacing(len(frame_files), max_frames_by_color)
            for idx in selected_indices:
                frame_path = os.path.join(path, frame_files[idx])
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                if resize_ratio != 1.0:
                    frame = cv2.resize(frame, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=-1)  # Garante que a imagem fique (H, W, 1)
                # Centraliza a imagem em um quadro quadrado
                width, height = 100, 100
                top = bottom = (height - frame.shape[0]) // 2
                left = right = (width - frame.shape[1]) // 2
                if (height - frame.shape[0]) % 2 != 0:
                    bottom += 1
                if (width - frame.shape[1]) % 2 != 0:
                    right += 1
                # Para grayscale, usar valor escalar para borda
                border_value = 0 if grayscale else [0, 0, 0]
                frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)
                if final_frame_shape != frame.shape[:2]:
                    frame = cv2.resize(frame, final_frame_shape, interpolation=cv2.INTER_AREA)
                    if grayscale and len(frame.shape) == 2:
                        frame = np.expand_dims(frame, axis=-1)
                if write_index:
                    idx_text = f"{color}_{idx}"
                    cv2.putText(frame, idx_text, (0, frame.shape[0] // 2), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1)
                frames.append(frame)
    return np.array(frames)

def load_all_data(indexes, colors, videos, base_path, resize_ratio, df_result,
                  num_originals_per_egg=NUM_ORIGINALS_PER_EGG, grayscale=False, augment=False, max_shift=5,
                  max_rotation=360, brightness_range=0.5, max_amount=None, num_augmentations=8, shuffle=True,
                  max_frames_by_color=MAX_FRAMES_BY_COLOR, pct_noise=0.02, shuffle_frames=True, additional_feature_list=None,
                  final_frame_shape=(100,100)):
    X_data, Y_data, X_additional_feature = [], [], []
    for egg_index in tqdm(indexes if max_amount is None else indexes[:max_amount], desc="Carregando dados"):
        try:
            for _ in range(num_originals_per_egg):
                original_frames = load_frames(egg_index, colors, videos, base_path, resize_ratio,
                                              grayscale=grayscale, max_frames_by_color=max_frames_by_color,
                                              final_frame_shape=final_frame_shape)
                obj_value = df_result[df_result['Índice'] == egg_index][OBJECTIVE_COLUMNS].iloc[0]
                additional_feature = (df_result[df_result['Índice'] == egg_index][additional_feature_list].iloc[0]
                                      if additional_feature_list else 0)
                X_data.append(original_frames)
                Y_data.append(obj_value)
                X_additional_feature.append(additional_feature)
                if augment:
                    for _ in range(num_augmentations):
                        augmented_frames = []
                        for frame in original_frames:
                            dx = np.random.randint(-max_shift, max_shift)
                            angulo = np.random.uniform(-max_rotation, max_rotation)
                            fator_brilho = np.random.uniform(1 - brightness_range, 1 + brightness_range)
                            h_flip = np.random.choice([True, False])
                            v_flip = np.random.choice([True, False])
                            rotate = np.random.choice([0, 90, 180, 270])
                            noise_factor = random.uniform(0, 1)
                            pct_noise_sample = noise_factor * pct_noise
                            apply_blur = np.random.choice([True, False])
                            shear_range = 0
                            swap_channels = np.random.choice([True, False])
                            augmented_frames.append(
                                augment_image(frame, dx, angulo, fator_brilho, h_flip, v_flip,
                                              rotate, pct_noise_sample, apply_blur, shear_range, swap_channels)
                            )
                        if shuffle_frames:
                            augmented_frames = random.sample(list(augmented_frames), len(augmented_frames))
                        X_data.append(np.array(augmented_frames))
                        Y_data.append(obj_value)
                        X_additional_feature.append(additional_feature)
        except Exception as e:
            print(f"Erro ao carregar ovo {egg_index}: {e}")
    if shuffle:
        data = list(zip(X_data, Y_data, X_additional_feature))
        random.shuffle(data)
        X_data, Y_data, X_additional_feature = zip(*data)
    return np.array(X_data), np.array(Y_data), np.array(X_additional_feature)

# ============================
# Modelo e Callbacks
# ============================
def create_model(num_frames, altura_frame, largura_frame, channels):
    input_shape = (num_frames, altura_frame, largura_frame, channels)
    input_layer = Input(shape=input_shape)
    x = TimeDistributed(Conv2D(16, (3, 3), activation='relu'))(input_layer)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = TimeDistributed(Conv2D(8, (3, 3), activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(len(OBJECTIVE_COLUMNS))(x)
    model = Model(inputs=input_layer, outputs=x)
    return model

class TestEvaluationCallback(Callback):
    def __init__(self, X_test, Y_test, objective_columns, interval=10):
        super(TestEvaluationCallback, self).__init__()
        self.X_test = X_test
        self.Y_test = Y_test
        self.objective_columns = objective_columns
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0 and epoch != 0:
            print(f"\nEvaluando no conjunto de teste na época {epoch}")
            send_telegram_message(f"Test metrics at epoch {epoch}")
            Y_pred = self.model.predict(self.X_test)
            for idx, col_name in enumerate(self.objective_columns):
                mae = mean_absolute_error(self.Y_test[:, idx], Y_pred[:, idx])
                mape = mean_absolute_percentage_error(self.Y_test[:, idx], Y_pred[:, idx])
                mse = mean_squared_error(self.Y_test[:, idx], Y_pred[:, idx])
                r2 = r2_score(self.Y_test[:, idx], Y_pred[:, idx])
                send_telegram_message(f"{col_name} MAE: {mae:.4f}, MAPE: {mape:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
            if r2 > 0.7:
                model_path = f'model_{epoch}.keras'
                self.model.save(model_path)
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_telegram_message(f"Modelo salvo com R2 maior que 0.7: {model_path} em {now_str}")
                send_telegram_file(model_path)

class SaveHistoryToExcelCallback(Callback):
    def __init__(self, filepath, save_freq=100):
        super(SaveHistoryToExcelCallback, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.history_data = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            self.history_data.setdefault(key, []).append(value)
        if (epoch + 1) % self.save_freq == 0:
            df = pd.DataFrame(self.history_data)
            df.to_excel(self.filepath, index=False)
    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.history_data)
        df.to_excel(self.filepath, index=False)

# ============================
# Função Principal
# ============================
def main():
    send_telegram_message(f"Treinamento iniciado em {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Carregar e processar planilha de dados
    df_result = pd.read_excel(EGG_DATA_SHEET_PATH)
    df_result = df_result[df_result["100% - Massa Total Calc/Aferida"] <= 0.02]
    df_result = df_result[["Índice"] + OBJECTIVE_COLUMNS].dropna()
    df_result = df_result[df_result['Índice'] != 476]
    print(f"Len of dataset: {len(df_result)}")
    print(df_result.head())
    
    # Selecionar índices válidos de ovos
    egg_indexes = []
    for egg_index_path in os.listdir(FRAMES_FOLDER_PATH):
        egg_index = int(egg_index_path.split('_')[1])
        if egg_index in df_result['Índice'].values:
            egg_indexes.append(egg_index)
    random.shuffle(egg_indexes)
    invalid_amount = len(os.listdir(FRAMES_FOLDER_PATH)) - len(egg_indexes)
    print(f"Valid eggs: {len(egg_indexes)}. Invalid eggs: {invalid_amount}.")
    
    # Dividir índices em treino, validação e teste
    total = len(egg_indexes)
    train_split = 0.6
    validation_split = 0.2
    test_split = 0.2
    train_indexes = egg_indexes[:int(total * train_split)]
    validation_indexes = egg_indexes[int(total * train_split):int(total * (train_split + validation_split))]
    test_indexes = egg_indexes[int(total * (train_split + validation_split)):]
    test_indexes.sort()
    print(f"Train: {len(train_indexes)}. Validation: {len(validation_indexes)}. Test: {len(test_indexes)}")
    
    # Carregar dados (originais e com augmentação)
    resize_ratio = 0.1
    X_train, Y_train, _ = load_all_data(train_indexes, COLORS, [1, 2], FRAMES_FOLDER_PATH,
                                        resize_ratio, df_result, num_augmentations=0,
                                        grayscale=GRAYSCALE, augment=True, max_frames_by_color=MAX_FRAMES_BY_COLOR,
                                        pct_noise=0.02, final_frame_shape=FINAL_FRAME_SHAPE, max_amount=2)
    X_val, Y_val, _ = load_all_data(validation_indexes, COLORS, [1, 2], FRAMES_FOLDER_PATH,
                                    resize_ratio, df_result, num_augmentations=0,
                                    grayscale=GRAYSCALE, augment=True, max_frames_by_color=MAX_FRAMES_BY_COLOR,
                                    pct_noise=0.02, final_frame_shape=FINAL_FRAME_SHAPE, max_amount=2)
    X_test, Y_test, _ = load_all_data(test_indexes, COLORS, [1, 2], FRAMES_FOLDER_PATH,
                                      resize_ratio, df_result, augment=False, shuffle=False,
                                      max_frames_by_color=MAX_FRAMES_BY_COLOR, final_frame_shape=FINAL_FRAME_SHAPE, max_amount=2)
    
    # Concatenar e separar novamente treino/validação
    new_X = np.concatenate((X_train, X_val), axis=0)
    new_Y = np.concatenate((Y_train, Y_val), axis=0)
    X_train, X_val, Y_train, Y_val = train_test_split(new_X, new_Y, test_size=0.2, random_state=seed_number)
    del new_X, new_Y
    
    print(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")
    
    # Augmentação individual dos dados de treino e validação
    X_train, Y_train = augment_image_individually(X_train, Y_train, max_shift=8, max_rotation=1,
                                                  brightness_range=0.5, num_augmentations=10, pct_noise=0.02)
    X_val, Y_val = augment_image_individually(X_val, Y_val, max_shift=8, max_rotation=1,
                                              brightness_range=0.5, num_augmentations=10, pct_noise=0.02)
    
    # Embaralhar a ordem dos frames em cada sample
    for i in range(len(X_train)):
        X_train[i] = random.sample(list(X_train[i]), len(X_train[i]))
    for i in range(len(X_val)):
        X_val[i] = random.sample(list(X_val[i]), len(X_val[i]))
    for i in range(len(X_test)):
        X_test[i] = random.sample(list(X_test[i]), len(X_test[i]))
    
    # Obter dimensões dos frames
    if GRAYSCALE:
        num_frames, altura_frame, largura_frame = X_train.shape[1:4]
        channels = 1
    else:
        num_frames, altura_frame, largura_frame, channels = X_train.shape[1:5]
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    
    # Salvar alguns frames para visualização
    sample_index = random.randint(0, len(X_train) - 1)
    output_frames_dir = 'frames_train'
    if os.path.exists(output_frames_dir):
        os.system(f'rm -rf {output_frames_dir}')
    os.makedirs(output_frames_dir, exist_ok=True)
    print(f"Salvando frames do sample {sample_index}")
    for i, image in enumerate(X_train[sample_index]):
        cv2.imwrite(os.path.join(output_frames_dir, f'{sample_index}_{i}.jpg'), image)
    
    # Criação e compilação do modelo
    model = create_model(num_frames, altura_frame, largura_frame, channels)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mape', metrics=['mape'])
    model.summary()
    
    # Definir callbacks
    model_checkpoint_path = f'model_best_total_mass_mae_sam_{"grayscale" if GRAYSCALE else "color"}.keras'
    model_checkpoint_callback = ModelCheckpoint(
        model_checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=100,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )
    history_path = f'model_history_{datetime.datetime.now().strftime("%d-%m-%Y_%H%M%S")}.xlsx'
    save_history_callback = SaveHistoryToExcelCallback(filepath=history_path, save_freq=10)
    test_evaluation_callback = TestEvaluationCallback(
        X_test=X_test,
        Y_test=Y_test,
        objective_columns=OBJECTIVE_COLUMNS,
        interval=25
    )
    
    # Treinamento do modelo
    batch_size = 50
    epochs = 2
    history = model.fit(
        X_train,
        Y_train,
        steps_per_epoch=int(len(X_train) / 4 / batch_size),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        validation_steps=int(len(X_val) / 4 / batch_size),
        callbacks=[model_checkpoint_callback, early_stopping_callback,
                   TensorBoard(log_dir='./logs', histogram_freq=1),
                   test_evaluation_callback, save_history_callback]
    )
    
    # Plotar e salvar gráfico de perda
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Gráfico de Aprendizado')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png')
    send_telegram_image('training_loss.png')
    pd.DataFrame(history.history).to_csv(f'training_history_mass_sam_{"grayscale" if GRAYSCALE else "color"}.csv', index=False)
    if os.path.isfile(history_path):
        send_telegram_file(history_path)
    
    if GRAYSCALE and X_test.shape[-1] != 1:
        X_test = X_test[..., :1]

    # Avaliação final no conjunto de teste
    performance = model.evaluate(X_test, Y_test, verbose=0)
    print(f"O modelo obteve {performance[1]:.4f} MAE no conjunto de teste.")
    send_telegram_message(f"O modelo obteve {performance[1]:.4f} MAE no conjunto de teste.")
    
    # Previsões e cálculo de métricas para cada variável de saída
    Y_pred = model.predict(X_test)
    result_dfs = {}
    for idx, col_name in enumerate(OBJECTIVE_COLUMNS):
        result_dfs[col_name] = pd.DataFrame({
            "real": Y_test[:, idx],
            "predict": Y_pred[:, idx],
            "difference": abs(Y_pred[:, idx] - Y_test[:, idx])
        })
    metrics = {}
    for col_name, df in result_dfs.items():
        mae = mean_absolute_error(df['real'], df['predict'])
        mape = mean_absolute_percentage_error(df['real'], df['predict'])
        mse = mean_squared_error(df['real'], df['predict'])
        r2 = r2_score(df['real'], df['predict'])
        metrics[col_name] = {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'R2': r2}
        send_telegram_message(f"{col_name}: MAE={mae:.2f}, MAPE={100*mape:.2f}%, MSE={mse:.2f}, R2={r2:.2f}")
    
    # Gerar gráficos de comparação entre predito e real para cada variável
    output_dir = "graficos_preditos_vs_reais"
    os.makedirs(output_dir, exist_ok=True)
    for col_name, df in result_dfs.items():
        plt.figure(figsize=(8, 6))
        plt.scatter(df['real'], df['predict'], alpha=0.7, label='Predito vs Real')
        min_val = min(df['real'].min(), df['predict'].min())
        max_val = max(df['real'].max(), df['predict'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (Y=X)')
        plt.title(f"Valores Preditos vs Reais para {col_name}")
        plt.xlabel("Valores Reais (Massa)")
        plt.ylabel("Valores Preditos (Massa)")
        plt.legend()
        plt.grid(True)
        file_path = os.path.join(output_dir, f"grafico_predito_vs_real_{col_name}.png")
        plt.savefig(file_path)
        send_telegram_image(file_path)
        plt.close()
        print(f"Gráfico salvo em: {file_path}")
    
    send_telegram_file(model_checkpoint_path, f"Modelo Treinado {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    send_telegram_message(str(model.summary()))

if __name__ == '__main__':
    main()
