import shutil
import os

f = open('label_map_k400.txt')
classes = []
for line in f.readlines():
    classes.append(line.strip())
print(classes)
f.close()
os.mkdir('dataset_400')
os.mkdir('dataset_400/train/')
for class_ in classes:
    # os.mkdir(f'dataset_400/train/{class_}')
    # os.system(f'cp -r dataset/train/{class_}/ dataset_400/train/')
    try: 
        shutil.copytree(f'dataset/train/{class_}', f'dataset_400/train/{class_}')
    except FileNotFoundError:
        print(f'{class_} not found')
        os.mkdir(f'dataset_400/train/{class_}')



