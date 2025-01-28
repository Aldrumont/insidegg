#Program to show the frames in a folder and show next frame if user type n
import cv2
import os
import numpy as np

def show_frames(frames_folder_path):
    frames_path = frames_folder_path
    files_list = os.listdir(frames_path)
    files_list = [file for file in files_list if file.endswith('.png')]
    files_list.sort()
    for file in files_list:
        frame = cv2.imread(os.path.join(frames_path, file))
        cv2.imshow('frame', frame)
        if cv2.waitKey(0) & 0xFF == ord('n'):
            continue
        else:
            break
    cv2.destroyAllWindows()
    return

# Example of usage
show_frames('frames_train')