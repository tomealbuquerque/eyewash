import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', default='data', help='path to the dataset')
parser.add_argument('--K', choices=[2], type=int, default=2)
parser.add_argument('--clean_images', type=int, default=500, help='number of clean cars images')
args = parser.parse_args()

#Imports
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pickle
import os
import cv2
from sklearn.utils import shuffle

# load images
X = []
Y = []

#images paths
dirname_clean = f'{args.datapath}/clean_cars'
dirname_dirty = f'{args.datapath}/dirty_cars'
paths_clean=[]
paths_dirty=[]
[paths_clean.append(f) for f in os.listdir(dirname_clean)]
[paths_dirty.append(f) for f in os.listdir(dirname_dirty)]


#Put images into arrays

#clean cars
for i in range(args.clean_images):
    img = cv2.imread(os.path.join(dirname_clean, paths_clean[i]), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))*255
    X.append(img)
    Y.append(0)

#dirty cars
for i in range(len(paths_dirty)):
    img = cv2.imread(os.path.join(dirname_dirty, paths_dirty[i]), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))*255
    X.append(img)
    Y.append(1)

X = np.array(X, np.uint8)
Y = np.array(Y, np.uint8)

X, Y = shuffle(X, Y)

# kfold
state = np.random.RandomState(1234)
kfold = StratifiedKFold(5, shuffle=True, random_state=state)
folds = [{'train': (X[tr], Y[tr]), 'test': (X[ts], Y[ts])} 
    for tr, ts in kfold.split(X, Y)]
pickle.dump(folds, open(f'data/k{args.K}.pickle', 'wb'))
