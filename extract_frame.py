import cv2
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from ultralytics import SAM
import torch
import gc

# Limpeza da memória CUDA
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

# Carrega o modelo SAM e exibe informações
model = SAM("sam2_t.pt")
model.info()

def generate_points(img, draw_points=False):
    """
    Gera pontos para segmentação com base nas dimensões da imagem.
    
    Args:
        img: Imagem em que os pontos serão gerados.
        draw_points: Se True, desenha os pontos na imagem (útil para debug).
    
    Returns:
        points: Lista de pontos [x, y] gerados.
    """
    img_width = img.shape[1]
    img_height = img.shape[0]
    x_center = img_width / 2
    y_center = img_height / 2

    left_limit = 0.3 * img_width
    right_limit = 0.8 * img_width

    points = []
    y_max = y_center + y_center / 1.5
    space_between_points = 100
    len_y_points = int(y_max / space_between_points)

    for i in range(len_y_points):
        y_coord = i * space_between_points
        points.append([x_center, y_coord])

        if i <= len_y_points / 2:
            num_side_points = i
        else:
            num_side_points = (len_y_points - i - 1) * 2

        for j in range(1, num_side_points // 2 + 1):
            x_right = x_center + j * space_between_points
            if x_right < right_limit:
                points.append([x_right, y_coord])
            x_left = x_center - j * space_between_points
            if x_left > left_limit:
                points.append([x_left, y_coord])

    if draw_points:
        for point in points:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 255, 255), -1)
    return points

def remove_background_with_sam(model, img, points, invert_mask=True):
    """
    Remove o fundo da imagem usando o modelo SAM com base nos pontos fornecidos.
    
    Args:
        model: Instância do modelo SAM.
        img: Imagem de entrada.
        points: Lista de pontos [x, y] pertencentes ao objeto.
        invert_mask: Se True, inverte o mask caso o resultado seja zerado (para evitar imagem totalmente transparente).
        
    Returns:
        masked_img: Imagem com fundo removido (canal alfa definido com base na máscara) ou None se falhar.
    """
    # Converte os pontos para array e adiciona dimensão de batch
    points_array = np.array(points).reshape(1, -1, 2)
    labels = np.array([1] * len(points)).reshape(1, -1)
    
    results = model(img, points=points_array, labels=labels, verbose=False)
    
    if results[0].masks is not None:
        # Obtém a máscara da primeira predição
        mask = results[0].masks.data[0].cpu().numpy()
        # Se a máscara estiver totalmente zerada, inverte-a (opcional)
        if invert_mask and np.sum(mask) == 0:
            mask = 1 - mask
        
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Se a máscara continuar zerada, emite aviso e retorna None
        if np.count_nonzero(mask_uint8) == 0:
            print("A máscara obtida está vazia, resultando em imagem transparente.")
            return None
        
        # Cria imagem com canal alfa
        b, g, r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        masked_img = cv2.merge([b, g, r, alpha])
        masked_img[:, :, 3] = mask_uint8
        
        return masked_img
    else:
        print("Nenhuma máscara encontrada pelo SAM.")
        return None

def list_all_files(root_folder):
    """
    Lista recursivamente todos os arquivos a partir de uma pasta.
    
    Args:
        root_folder: Caminho raiz.
    
    Returns:
        Lista de caminhos completos dos arquivos.
    """
    all_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files

def read_and_extract_frames(egg_data_table_path, color_data_table_path, colors, frames_per_color, num_eggs,
                            selection_mode='sequential', specific_videos=None, offset=10, cut_frame_dict=None,
                            min_index=0, max_index=1000, videos_folder_path='videos', frames_folder_path='frames',
                            points=None, save_original=False):
    """
    Lê tabelas de dados, extrai frames de vídeos e aplica remoção de fundo com SAM.
    
    Args:
        egg_data_table_path: Caminho para a tabela de ovos.
        color_data_table_path: Caminho para a tabela de cores.
        colors: Lista de cores a serem extraídas.
        frames_per_color: Número de frames a extrair por cor.
        num_eggs: Número de ovos a processar.
        selection_mode: 'random', 'sequential' ou 'specific'.
        specific_videos: Lista de índices específicos (se selection_mode for 'specific').
        offset: Offset para ajuste dos índices dos frames.
        cut_frame_dict: Dicionário com chaves 'min_height', 'max_height', 'min_width', 'max_width' para corte.
        min_index, max_index: Intervalo de índices de ovos.
        videos_folder_path: Pasta onde os vídeos estão.
        frames_folder_path: Pasta onde os frames serão salvos.
        points: Se None, utiliza os pontos gerados pela função generate_points.
        save_original: Se True, salva os frames originais (sem pontos desenhados); default False.
    
    Retorna:
        extracted_frames: Dicionário com informações dos frames extraídos.
    """
    # Carrega as tabelas
    df_egg_data = pd.read_excel(egg_data_table_path)
    df_color_data = pd.read_excel(color_data_table_path).dropna()
    df_color_data_columns = list(df_color_data.columns)
    colors = [color.lower() for color in colors]
    
    # Seleção dos ovos conforme o modo
    if selection_mode == 'random':
        df_egg_data = df_egg_data[(df_egg_data['Índice'] >= min_index) & (df_egg_data['Índice'] <= max_index)]
        df_egg_data = df_egg_data.sample(min(num_eggs, len(df_egg_data)))
    elif selection_mode == 'specific' and specific_videos is not None:
        df_egg_data = df_egg_data[df_egg_data['Índice'].isin(specific_videos)]
    else:  # Padrão sequencial
        df_egg_data = df_egg_data.head(num_eggs)
    
    extracted_frames = {}
    df_egg_data = df_egg_data.sort_values(by=['Índice'])
    
    for _, row in tqdm(df_egg_data.iterrows(), total=df_egg_data.shape[0], desc="Processando ovos"):
        egg_index = int(row['Índice'])
        print(f"Egg index: {egg_index}")
        videos_indexes = [int(row['video A']), int(row['video B'])]
        extracted_frames[egg_index] = {}
        for video_index in videos_indexes:
            video_path = os.path.join(videos_folder_path, f'video_{video_index}.mp4')
            extracted_frames[egg_index][video_index] = {}
            
            if not os.path.exists(video_path):
                print(f"Vídeo não encontrado: {video_path}")
                continue
            
            cap = cv2.VideoCapture(video_path)
            for color in colors:
                current_data = df_color_data.loc[df_color_data['video_index'] == video_index]
                if current_data.empty:
                    continue
                start_frame = int(current_data[f'{color}_start'].values[0]) + offset
                current_column_index = df_color_data_columns.index(f'{color}_start')
                next_column_name = df_color_data_columns[current_column_index + 1]
                end_frame = int(current_data[next_column_name].values[0]) - offset
                
                frame_indices = np.linspace(start_frame, end_frame, frames_per_color, endpoint=False, dtype=int)
                
                for frame_idx in frame_indices:
                    out_dir = os.path.join(frames_folder_path, f'egg_{egg_index}', f'video_{video_index}', color)
                    os.makedirs(out_dir, exist_ok=True)
                    final_path = os.path.join(out_dir, f'frame_{frame_idx}.png')
                    # Pasta para frames originais (sem alteração) – será salvo apenas se save_original for True
                    original_dir = os.path.join("frames_original", f'egg_{egg_index}', f'video_{video_index}', color)
                    os.makedirs(original_dir, exist_ok=True)
                    final_path_original = os.path.join(original_dir, f'frame_{frame_idx}.png')
                    
                    # Se o frame já foi processado, pula
                    if final_path in files_list:
                        continue
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Frame {frame_idx} não lido no vídeo {video_index}.")
                        continue
                    
                    # Aplica o corte do frame, se definido
                    if cut_frame_dict:
                        frame = frame[cut_frame_dict['min_height']:cut_frame_dict['max_height'],
                                      cut_frame_dict['min_width']:cut_frame_dict['max_width']]
                    
                    # Gera os pontos automaticamente (sem desenhar na imagem original)
                    seg_points = generate_points(frame, draw_points=False)
                    
                    # Salva o frame original apenas se a opção save_original for True
                    if save_original:
                        cv2.imwrite(final_path_original, frame)
                    
                    # Aplica a remoção de fundo usando SAM
                    processed_frame = remove_background_with_sam(model, frame, seg_points, invert_mask=True)
                    if processed_frame is None:
                        print(f"Falha na remoção de fundo para egg {egg_index}, vídeo {video_index}, frame {frame_idx}.")
                        continue
                    cv2.imwrite(final_path, processed_frame)
            cap.release()
    return extracted_frames

# Parâmetros de background para diferentes cores LED
colors_background_parameters_dict = {
    "white": {  
        "hmin": 0, "hmax": 360, "smin": 0, "smax": 360, "vmin": 40, "vmax": 100
    },
    "red": {
        "hmin": 0, "hmax": 360, "smin": 0, "smax": 360, "vmin": 65, "vmax": 100
    },
    "blue": {
        "hmin": 0, "hmax": 360, "smin": 0, "smax": 360, "vmin": 0, "vmax": 100
    },
    "green": {
        "hmin": 0, "hmax": 360, "smin": 0, "smax": 360, "vmin": 10, "vmax": 100
    }
}

# Configurações gerais
colors_to_extract = ['blue', 'red', 'white', 'green']
frames_per_color = 36  # Número de frames por cor
num_eggs = 1000
selection_mode = 'random'  # 'random', 'sequential' ou 'specific'
cut_frame_dict = {'min_height': 75, 'max_height': 950, 'min_width': 550, 'max_width': 1500}

# Lista de arquivos já processados para evitar redundância
files_list = list_all_files('frames')

# Executa a extração e processamento dos frames
read_and_extract_frames(
    egg_data_table_path='dados_ovos.xlsx',
    color_data_table_path='color_start_times.xlsx',
    colors=colors_to_extract,
    frames_per_color=frames_per_color,
    num_eggs=num_eggs,
    selection_mode=selection_mode,
    cut_frame_dict=cut_frame_dict,
    min_index=0,
    max_index=1000,
    videos_folder_path='videos',
    frames_folder_path='frames',
    specific_videos=None,
    offset=20,
    points=None,           # Usa pontos gerados automaticamente
    save_original=False    # Por padrão, não salva os frames originais
)
