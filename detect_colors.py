import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
from tqdm import tqdm

def get_avg_rgb_values(rgb_deque):
    return np.array(rgb_deque).mean(axis=0)

def get_sum_rgb_values(frame, divide_by_area=True):
    b_sum, g_sum, r_sum = np.sum(frame, axis=(0, 1))
    if divide_by_area:
        area = frame.shape[0] * frame.shape[1]
        r_sum, g_sum, b_sum = r_sum / area, g_sum / area, b_sum / area
    return r_sum, g_sum, b_sum

def detect_led_color(rgb_avg, frame_count):
    r_avg, g_avg, b_avg = rgb_avg
    if frame_count < 250:
        return 'White'
    if r_avg > g_avg + b_avg:
        return 'Red'
    elif g_avg > r_avg + b_avg:
        return 'Green'
    elif b_avg > r_avg + g_avg:
        return 'Blue'
    elif r_avg + g_avg + b_avg < 10:
        return 'Black'
    else:
        return 'White'

def transform_to_dict_with_longest_period(colors):
    color_ranges = {}
    current_color = colors[0]
    start_index = 0

    for i in range(1, len(colors)):
        if colors[i] != current_color:
            end_index = i - 1
            period_length = end_index - start_index
            if current_color not in color_ranges or period_length > (color_ranges[current_color][1] - color_ranges[current_color][0]):
                color_ranges[current_color] = (start_index, period_length)
            current_color = colors[i]
            start_index = i

    end_index = len(colors) - 1
    period_length = end_index - start_index
    if current_color not in color_ranges or period_length > (color_ranges[current_color][1] - color_ranges[current_color][0]):
        color_ranges[current_color] = (start_index, period_length)

    return {color: start for color, (start, _) in color_ranges.items()}

def process_video(video_path, watch_video=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    rgb_deque = deque(maxlen=10)
    led_colors = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc=f'Processing {video_path}', leave=False):
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[75:800, 550:1500]
        rgb_sum = get_sum_rgb_values(frame)
        rgb_deque.append(rgb_sum)

        if len(rgb_deque) == 10:
            rgb_avg = get_avg_rgb_values(rgb_deque)
            led_color = detect_led_color(rgb_avg, frame_count)
            led_colors.append(led_color)

        if watch_video:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_count += 1

    cap.release()
    return transform_to_dict_with_longest_period(led_colors), total_frames

def process_all_videos(videos_path, watch_video=False, limit=None, randomize=False):
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    results = []

    if randomize:
        np.random.shuffle(video_files)
    if limit:
        video_files = video_files[:limit]
    
    for video_file in video_files:
        video_path = os.path.join(videos_path, video_file)
        color_ranges, total_frames = process_video(video_path, watch_video)
        row = [video_file] + [color_ranges.get(color) for color in ['White', 'Red', 'Blue', 'Green']] + [total_frames]
        results.append(row)

    return pd.DataFrame(results, columns=['video_path', 'white_start', 'red_start', 'blue_start', 'green_start', 'total_frames'])

# Opção para assistir aos vídeos durante o processamento
watch_video_option = False

# Processar todos os vídeos na pasta e gerar a tabela
result_table = process_all_videos('videos', watch_video=watch_video_option, limit=None, randomize=False)
result_table.to_excel("color_start_times.xlsx", index=False)

cv2.destroyAllWindows()
