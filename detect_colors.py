import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
from tqdm import tqdm
import concurrent.futures
from itertools import groupby

# -----------------------------------------------------------------------------
# Use cv2.mean to quickly compute the average (mean) RGB values for a region.
def get_mean_rgb_values(frame):
    # Crop frame has been already applied.
    # cv2.mean returns (B, G, R, [alpha]) so we re-order to (R, G, B)
    mean_vals = cv2.mean(frame)
    return mean_vals[2], mean_vals[1], mean_vals[0]

# -----------------------------------------------------------------------------
# Decide the LED color based on the averaged RGB and the frame count.
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

# -----------------------------------------------------------------------------
# Using groupby to find the start index of the longest contiguous period for each color.
def transform_to_dict_with_longest_period(colors):
    longest_runs = {}
    for color, group in groupby(enumerate(colors), key=lambda x: x[1]):
        group_list = list(group)
        start = group_list[0][0]
        run_length = len(group_list) - 1  # same as original: period = end_index - start_index
        if color not in longest_runs or run_length > longest_runs[color][1]:
            longest_runs[color] = (start, run_length)
    # Return a dict mapping color to the start index of its longest period.
    return {color: start for color, (start, _) in longest_runs.items()}

# -----------------------------------------------------------------------------
# Process a single video. Note the optimizations:
#  - We pre-define the cropping region.
#  - We maintain a running sum for the sliding window.
def process_video(video_path, watch_video=False):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    # Use a deque for the sliding window; we also maintain a running sum to avoid
    # recreating a NumPy array every time.
    window_size = 10
    sliding_window = deque(maxlen=window_size)
    sliding_sum = np.zeros(3, dtype=np.float64)  # holds cumulative (R, G, B)
    led_colors = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Pre-define the crop slice (this is the same every frame)
    crop_region = (slice(75, 800), slice(550, 1500))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame once.
        frame = frame[crop_region]
        
        # Get the average RGB of the cropped frame using cv2.mean.
        new_rgb = get_mean_rgb_values(frame)
        
        # Update our sliding window and running sum.
        if len(sliding_window) < window_size:
            sliding_window.append(new_rgb)
            sliding_sum += np.array(new_rgb)
        else:
            oldest = sliding_window.popleft()
            sliding_sum -= np.array(oldest)
            sliding_window.append(new_rgb)
            sliding_sum += np.array(new_rgb)
        
        # Once we have a full window, compute the sliding average.
        if len(sliding_window) == window_size:
            rgb_avg = sliding_sum / window_size
            led_color = detect_led_color(rgb_avg, frame_count)
            led_colors.append(led_color)

        if watch_video:
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        frame_count += 1

    cap.release()
    return transform_to_dict_with_longest_period(led_colors), total_frames

# -----------------------------------------------------------------------------
# Process all videos in the given directory.
# If watch_video is False, you can enable parallel processing.
def process_all_videos(videos_path, watch_video=False, limit=None, randomize=False, parallel=True):
    video_files = [f for f in os.listdir(videos_path) if f.endswith('.mp4')]
    
    if randomize:
        np.random.shuffle(video_files)
    if limit:
        video_files = video_files[:limit]
    
    results = []
    video_paths = [os.path.join(videos_path, vf) for vf in video_files]
    
    # If you are watching the video, it is best to process serially.
    if parallel and not watch_video:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all the video processing tasks.
            future_to_video = {executor.submit(process_video, vp, watch_video): vp for vp in video_paths}
            for future in tqdm(concurrent.futures.as_completed(future_to_video),
                               total=len(future_to_video), desc="Processing videos"):
                video_path = future_to_video[future]
                try:
                    color_ranges, total_frames = future.result()
                    video_file = os.path.basename(video_path)
                    video_index = int(video_file.split('.')[0].replace('video_', ''))
                    row = [video_index] + [color_ranges.get(color) for color in ['White', 'Red', 'Blue', 'Green']] + [total_frames]
                    results.append(row)
                except Exception as exc:
                    print(f"Exception processing {video_path}: {exc}")
    else:
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(videos_path, video_file)
            color_ranges, total_frames = process_video(video_path, watch_video)
            row = [video_file] + [color_ranges.get(color) for color in ['White', 'Red', 'Blue', 'Green']] + [total_frames]
            results.append(row)
    
    return pd.DataFrame(results, columns=['video_path', 'white_start', 'red_start', 'blue_start', 'green_start', 'total_frames'])

# -----------------------------------------------------------------------------
# Main execution block.
if __name__ == '__main__':
    # Set to True only if you want to see the video playback (which slows processing).
    watch_video_option = False
    
    # When not watching, enable parallel processing.
    parallel_process = not watch_video_option
    
    # Process all videos in the 'videos' folder.
    result_table = process_all_videos('videos', watch_video=watch_video_option,
                                      limit=None, randomize=False,
                                      parallel=parallel_process)
    # Save the result to an Excel file.
    result_table.to_excel("color_start_times.xlsx", index=False)
    
    if watch_video_option:
        cv2.destroyAllWindows()
