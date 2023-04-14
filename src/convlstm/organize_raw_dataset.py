import json
import os
import shutil
import itertools
import mediapipe.config as config
import os

from mediapipe.config import SPLITS, EXTENSION

def rm_error_info(func, path, _):
    print("INFO: The path", path, "does not exist. Skipping...")


def get_n_gloss(indexfile='data/WLASL_v0.3.json'):
    content = json.load(open(indexfile))
    return len([items for items in content])


def remove_original_videos():
    shutil.rmtree(os.path.join(config.VIDEOS_PATH, "*.mp4"), onerror=rm_error_info)


def remove_empty_folders(root):
    folders = list(os.walk(root))[1:]

    for folder in folders:
        # folder example: ('FOLDER/3', [], ['file'])
        if not folder[2]:
            shutil.rmtree(folder[0])


def gloss_ranking(content, vid_directory, top_k):
    ranking = {}
    for entry in content:
        gloss = entry['gloss']
        
        appereances = []
        for split in SPLITS:
            path_to_check = os.path.join(vid_directory, split, gloss)
            # If there is a gloss that does not appear in a split, we skip it
            try:
                items = os.listdir(path_to_check)
                appereances.append(len(items))
            except OSError:
                appereances.append(0)

        # we add up all the videos from each split, given a gloss
        ranking[gloss] = sum(appereances)

    # sorting the dictionary based on the value
    top_ranking = {k: v for k, v in sorted(ranking.items(), key=lambda item: item[1], reverse=True)}

    return dict(itertools.islice(top_ranking.items(), top_k))


def organize(indexfile='data/WLASL_v0.3.json', vid_directory='data/videos', top_k = 1000):
    if indexfile == 'nil':
        print('No index specified. Exiting.')
        return

    content = json.load(open(indexfile))

    if top_k > get_n_gloss(indexfile):
        print("The number of the top_k is greater the total glosses of the dataset")
        return
    
    gloss_rank = gloss_ranking(content, vid_directory, top_k)
    print("Ranking created with top", top_k, "glosses/labels...")

    for entry in content:
        gloss = entry['gloss']
        instances = entry['instances']

        # if the gloss is in the top_k, then we add it to the top_k dataset
        if gloss in gloss_rank.keys():
            for inst in instances:
                vid_id = inst['video_id']
                split = inst['split']

                source = os.path.join(vid_directory, vid_id+EXTENSION)
                destination = os.path.join("./data", f"top_{top_k}", split, gloss)

                # create the dataset structure /data/videos/top_k/<train|test|val>/gloss
                if not os.path.exists(destination): 
                    os.makedirs(destination)
                
                # and now, we copy from /data/videos to /data/videos/top_k/<train|test|val>/gloss
                if os.path.exists(source):
                    shutil.copy(source, destination)
    

def create_datasets(indexfile='data/WLASL_v0.3.json', 
                      vid_directory='data/videos', 
                      keep_original=True):
    
    '''
    This function creates a preset of datasets that contain:
    * top_10: dataset with the top 10 datasets
    * top_100: dataset with the top 100 datasets
    * top_200: dataset with the top 200 datasets
    * top_500: dataset with the top 500 datasets
    * top_1000: dataset with the top 1000 datasets
    * top_2000: dataset with the top 2000 datasets
    '''
    # sizes = [10, 100, 200, 500, 1000, 2000]
    sizes = [10, 100, 200]
    for K in sizes:
        organize(indexfile, vid_directory, top_k=K)

    if keep_original == False:
        remove_original_videos()

    # There might be some glosses where there are no videos. We will erase them to have some consistency.
    # in our dataset.
    for K in sizes:
        for sp in SPLITS:
            root = f'./data/top_{K}/{sp}/'
            remove_empty_folders(root)

    return

if __name__ == "__main__":
    create_datasets()