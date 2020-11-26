'''
This script works independent and it helps in grabbing
all the smaller pickle files (saved as dictionaries) for PA and we use
all those dictionaries and make them into one large dictionary
'''
import pickle
import os
PATH_DIR = 'images_pickled/all_PA_files/'
all_PA_img = dict()
count_keys = 0
keys = []
for file_name in os.listdir(PATH_DIR):
  full_name = PATH_DIR + file_name
  with open(full_name,'rb') as f:
    dict_img = pickle.load(f)
    cur_keys = list(dict_img.keys())
    keys += list(dict_img.keys())
    all_PA_img.update(dict_img)

print(len(set(keys)))
with open(f"{PATH_DIR}PA_images.pkl",'wb') as f:
  pickle.dump(all_PA_img,f)
