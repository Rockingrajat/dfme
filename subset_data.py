import random
import os
import shutil
os.mkdir('dataset_600_small')
os.mkdir('dataset_600_small/train/')
classes = os.listdir('dataset/train')
for class_ in classes:
    all_videos = os.listdir(f'dataset/train/{class_}')
    if len(all_videos) >= 10:
        videos = all_videos[:10]
    else:
        videos = all_videos
    #videos = os.listdir(f'dataset_400/train/{class_}')[:10]
    os.mkdir(f'dataset_600_small/train/{class_}')
    for video in videos:
        shutil.copy(f'dataset/train/{class_}/{video}', f'dataset_600_small/train/{class_}/')
