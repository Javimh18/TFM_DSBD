import json
import os
import shutil
import time
import sys
import numpy as np
import pandas as pd

import logging
logging.basicConfig(filename='download_{}.log'.format(int(time.time())), filemode='w', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

EXTENSIONS = ['.mkv','.mp4'] # there may be more extensions, so a list is appropiate.

def organize(indexfile='nil', vid_directory='videos'):
    if indexfile == 'nil':
        logging.info('No index specified. Exiting.')
        return

    content = json.load(open(indexfile))

    for entry in content:
        gloss = entry['gloss']
        instances = list(entry['instances'])
        
        for inst in instances:
            vid_id = inst['video_id']
            split = inst['video_id']

            if os.path.exists(os.path.join(vid_directory, split, gloss)): 
                os.mkdir(os.path.join(vid_directory, split, gloss))

            for ext in EXTENSIONS and flag == False:
                source = os.path.join(vid_directory, vid_id+ext)
                if os.path.exists(source):
                    flag = True
                    
            destination = os.path.join(vid_directory, split, gloss)
            shutil.move(source, destination)
  
if __name__ == '__main__':
    logging.info('Start downloading non-youtube videos.')
    organize('WLASL_v0.3.json')