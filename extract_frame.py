import cv2
import pandas as pd
import numpy as np
import os
import random
from tqdm import tqdm
from ultralytics import SAM
import torch
import gc

# Limpeza da memória CUDA
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

# Carrega o modelo SAM
model = SAM("sam2_t.pt")
model.info()

def generate_points(img, draw_points=False):
    """
    Gera os pontos para segmentação com base nas dimensões da imagem.
    
    Args:
        img: Imagem em que os pontos serão gerados.
    
    Returns:
        points: Lista de pontos gerados.
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
            num_side_points = i * 1
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


def read_and_extract_frames(egg_data_table_path, color_data_table_path, colors, frames_per_color, num_eggs, selection_mode='sequential',
                            specific_videos=None, offset=10, cut_frame_dict=None, min_index=0, max_index=1000, videos_folder_path='videos',
                            frames_folder_path='frames', points='center'):

    # Load the table
    df_egg_data = pd.read_excel(egg_data_table_path)
    df_color_data = pd.read_excel(color_data_table_path)
    df_color_data = df_color_data.dropna()

    colors = [color.lower() for color in colors]
    # Determine the eggs to process based on the selection mode
    if selection_mode == 'random':
        df_egg_data = df_egg_data[df_egg_data['Índice'] >= min_index]
        df_egg_data = df_egg_data[df_egg_data['Índice'] <= max_index]
        df_egg_data = df_egg_data.sample(min(num_eggs, len(df_egg_data)))
    elif selection_mode == 'specific' and specific_videos is not None:
        df_egg_data = df_egg_data[df_egg_data['Índice'].isin(specific_videos)]
    else:  # Default to sequential if mode is not recognized or specific_videos is not provided
        df_egg_data = df_egg_data.head(num_eggs)

    # Initialize a dictionary to store the frames
    extracted_frames = {}
    df_color_data_columns = list(df_color_data.columns)

    #sort the df_color_data by video_index
    df_egg_data = df_egg_data.sort_values(by=['Índice'])

    for _, row in tqdm(df_egg_data.iterrows(), total=df_egg_data.shape[0]):
        egg_index = int(row['Índice'])
        print(f"Egg index: {egg_index}")
        videos_indexes = [int(row['video A']), int(row['video B'])]
        extracted_frames[egg_index] = {}
        for video_index in videos_indexes:
            video_path = os.path.join(videos_folder_path, f'video_{video_index}.mp4')
            extracted_frames[egg_index][video_index] = {}



            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                for color in colors:
                    current_data = df_color_data.loc[df_color_data['video_index'] == video_index]
                    if len(current_data) == 0:
                        continue
                    start_frame = int(current_data[f'{color}_start'].values[0]) + offset
                    # get the name of the next column
                    current_column_index = df_color_data_columns.index(f'{color}_start')
                    next_column_name = df_color_data_columns[current_column_index + 1]
                    end_frame = int(current_data[next_column_name].values[0]) - offset

                    frame_indices = np.linspace(start_frame, end_frame, frames_per_color, endpoint=False, dtype=int)

                    color_frames = []
                    for frame_idx in frame_indices:
                        final_path = os.path.join(frames_folder_path, f'egg_{egg_index}', f'video_{video_index}', color, f'frame_{frame_idx}.png')
                        final_path_no_remove = os.path.join("frames_original", f'egg_{egg_index}', f'video_{video_index}', color, f'frame_{frame_idx}.png')
                        if final_path not in files_list:
                            os.makedirs(os.path.dirname(final_path), exist_ok=True)
                            os.makedirs(os.path.dirname(final_path_no_remove), exist_ok=True)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            # if not ret:
                                # break
                            frame = frame[cut_frame_dict['min_height']:cut_frame_dict['max_height'],
                                            cut_frame_dict['min_width']:cut_frame_dict['max_width']]
                            if points == 'center':
                                points = [[frame.shape[1] // 2, frame.shape[0] // 2]]
                            if not points:
                                copy_frame = frame.copy()
                                points = generate_points(copy_frame, draw_points=True)
                            cv2.imwrite(final_path_no_remove, frame)
                            # frame = remove_background(frame, led_color=color.lower())
                            frame = remove_background_with_sam(model, frame, points)
                                


                            cv2.imwrite(final_path, frame)


                cap.release()

    return extracted_frames

def remove_background(img, hmin=0, hmax=360, smin=0, smax=360, vmin=0, vmax=100, led_color=None):
    # Load the image

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if not led_color:
        # Define the color range for the background
        lower_range = np.array([hmin/2, smin*2.55, vmin*2.55])
        upper_range = np.array([hmax/2, smax*2.55, vmax*2.55])
    else:
        lower_range = np.array([colors_background_parameters_dict[led_color]['hmin']/2, colors_background_parameters_dict[led_color]['smin']*2.55, colors_background_parameters_dict[led_color]['vmin']*2.55])
        upper_range = np.array([colors_background_parameters_dict[led_color]['hmax']/2, colors_background_parameters_dict[led_color]['smax']*2.55, colors_background_parameters_dict[led_color]['vmax']*2.55])

    # Threshold the image to remove the background
    mask = cv2.inRange(hsv, lower_range, upper_range)
    img = cv2.bitwise_and(img, img, mask=mask)

    return img

def remove_background_with_sam(model, img, points):
    """
    Remove o fundo da imagem usando o modelo SAM, aplicando a segmentação com base nos pontos fornecidos.
    
    Args:
        model: Instância do modelo SAM.
        img: Imagem de entrada.
        points: Lista de pontos fornecidos para o modelo SAM.
        
    Returns:
        masked_img: A imagem com o fundo removido (somente o objeto segmentado) e fundo transparente.
    """
    results = model(img, points=points, labels=[1], verbose=False)

    if results[0].masks is not None:
        # Obtém a máscara da primeira previsão do SAM
        mask = results[0].masks.data[0].cpu().numpy()

        # Converte a máscara para uint8 e escala para 255
        mask_uint8 = (mask * 255).astype(np.uint8)

        # Adiciona um canal alfa à imagem
        b, g, r = cv2.split(img)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        masked_img = cv2.merge([b, g, r, alpha])

        # Aplica a máscara para definir a transparência
        masked_img[:, :, 3] = mask_uint8

        return masked_img
    else:
        print("Nenhuma máscara encontrada.")
        return None


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

colors_to_extract = ['blue','red','white','green']
frames_per_color = 36 # 360/40 = 9 frames per color
num_eggs = 1000
selection_mode = 'random'  # Can be 'random', 'sequential', or 'specific'
cut_frame_dict = {'min_height': 75, 'max_height': 950, 'min_width': 550, 'max_width': 1500}

import os

def list_all_files(root_folder):
    all_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


frames_folder_path = 'frames'
files_list = list_all_files(frames_folder_path)


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
    frames_folder_path=frames_folder_path,
    specific_videos=None,
    offset=20,
)

