from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import json
from tqdm import tqdm
import os
import random 
from joblib import Parallel, delayed

'''
{
    "$name": {
        "annotations": {
            "label": "$label",
            "segment": [
                17.0,
                27.0
            ]
        },
        "duration": 10.0,
        "subset": "train/test/val",
        "url": "$url"
    }
}
'''
def justCheckExistence(videourl):
    try:
        yt = YouTube("https://www.youtube.com/watch?v="+videourl)
        yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    except:
        return 0
    return 1

def get_diff():
    f = open('kin_600_label.txt')
    classes = []
    for line in f.readlines():
        classes.append(line.strip())
    f1 = open('label_map_k400.txt')
    classes_400 = []
    for line in f1.readlines():
        classes_400.append(line.strip())

    set_classes = set(classes)
    set_classes_400 = set(classes_400)
    set_data = set(os.listdir('dataset/train'))

    # print(set_data - set_classes)
    # print(set_classes - set_data)
    # print(set_classes_400.issubset(set_classes))
    return (set_classes_400-set_classes)

def downloadYouTube(videourl, path, name, segment):
    try:
        # many videos are not available
        yt = YouTube("https://www.youtube.com/watch?v="+videourl)
        yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    except:
        return None, None
    tempadd = yt.download('')
    ffmpeg_extract_subclip(tempadd, segment[0], segment[1], targetname=os.path.join(path,name))
    os.remove(tempadd)
    return name, os.path.join(path, name)

def downloadLabel(numDownTillNow, classWiseSampReq, classWiseKeys, label, pathDownload, datafull, validSamples):
    numValid = numDownTillNow.get(label, 0)
    ci = 0
    while numValid<classWiseSampReq[label] and ci<len(classWiseKeys[label]):
        key = classWiseKeys[label][ci]
        # if justCheckExistence(key)==1:
        #     validSamples[key] = datafull[key]
        #     numValid+=1
        name = label+'_'+key+'_gb.mp4'
        if os.path.exists(os.path.join(pathDownload, name)):
            ci+=1
            continue
        segment = datafull[key]['annotations']['segment']
        namevid, pathvid = downloadYouTube(key, pathDownload, name, segment)
        if namevid!=None and pathvid!=None:
            validSamples[key] = datafull[key]
            numValid+=1
        ci+=1

    return

def main():
    random.seed(1)
    dataset_json_path = 'kinetics600_train.json'
    root = 'dataset/train/'

    if not os.path.exists(root):
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        os.mkdir(root)
    
    datasetSize = 0.05
    with open(dataset_json_path) as f:
        datafull = json.load(f)
    allVideos = datafull.keys()
    print('Total samples: ', len(list(allVideos)))
    classWiseKeys = {}
    for key in tqdm(allVideos):
        label = datafull[key]['annotations']['label']
        if classWiseKeys.get(label):
            classWiseKeys[label].append(key)
        else:
            classWiseKeys[label] = [key]
    # print(classWiseKeys)
    classWiseSampReq = {}
    currIndex = {} # in case of any failure
    for label in classWiseKeys.keys():
        random.shuffle(classWiseKeys[label])
        classWiseSampReq[label] = int(len(classWiseKeys[label])*datasetSize)
        currIndex[label] = 0
    
    validSamples = {}
    numDownTillNow = {}
    for key in classWiseKeys.keys():
        if os.path.exists(os.path.join(root, key)):
            numDownTillNow[key] = len(os.listdir(os.path.join(root, key)))
        else:
            os.mkdir(os.path.join(root, key))
            numDownTillNow[key] = 0

    Parallel(n_jobs=4)(delayed(downloadLabel)(numDownTillNow, classWiseSampReq, classWiseKeys, label, os.path.join(root, label), datafull, validSamples) for label in tqdm(classWiseKeys.keys()))
    
    return 


if __name__=='__main__':
    main()