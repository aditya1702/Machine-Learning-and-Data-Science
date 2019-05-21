#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 23:10:56 2018

@author: Vedant Choudhary and Aditya Vyas
@affiliation: Rutgers University, New Brunswick
"""

####################################################################
#################### create_pickle_data.py #########################
###  create_pickle_data.py creates pickle data which will be     ###
###  loaded to be provided as input and output value for the     ###
###  CRNN model. Pickle basically serializes the data in binary  ###
###  format, saving up on space.                                 ###
####################################################################
####################################################################

# Importing the required libraries for operations in the code
import os
import pandas as pd
import ast
from pickle import dump
import numpy as np
from util import METADATA_DIR, AUDIO_DIR, DATA_DIR, TRACK_COUNT, MAIN_DIR, PICKLE_DIR, GENRES, load_track
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
import util
import csv


# The tracks data is cleaned and correctly organized. This dataset will have multi-class labels.
# So, in order to access suppose two columns such as track and it's top_genre, you will have to
# use multilabels like tracks['track', 'genre_top']
# Input - filename which is the destination to where tracks.csv is stored
# Return - cleaned up tracks dataset
def cleanTracksData(filename):
    tracks = pd.read_csv(filename, index_col = 0, header=[0, 1])

    COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres')]
    for column in COLUMNS:
        tracks[column] = tracks[column].map(ast.literal_eval)

    COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
    for column in COLUMNS:
        tracks[column] = pd.to_datetime(tracks[column])

    SUBSETS = ('small', 'medium', 'large')
    tracks['set', 'subset'] = tracks['set', 'subset'].astype('category', categories=SUBSETS, ordered=True)

    COLUMNS = [('track', 'license'), ('artist', 'bio'), ('album', 'type'), ('album', 'information')]
    for column in COLUMNS:
        tracks[column] = tracks[column].astype('category')

    return tracks

# HARD-CODED THE GENRES TO COMMONS.PY FOR EASY USAGE, if someone decides to use some other
# dataset of fma, they can use this function to retrieve genres. Change small to medium/large
# Input - tracks and genres dataset
# Return - list of genres to be predicted (present in small subset of data)
#def validGenres(tracks, genres):
#    subset = tracks['set', 'subset'] <= "small"
#    d = genres.reset_index().set_index('title')
#    d = d.loc[tracks.loc[subset, ('track', 'genre_top')].unique()]
#
#    return list(d.index)

# Valid genres present in the subset "small". Total 8 in number
# One Hot Encoding done and stored in a dictionary
#GENRES = validGenres(tracks, genres_data)
#GENRES = sorted(GENRES)

# This function gets all the tracks present in fma_small dataset by traversing through it,
# along with its genres.
# Input - Audio directory and tracks dataset
# Return - a list [track_number, track_path, track_genre]
def getTrackIDs(aud_dir, tracks):
    track_ids = []

    small = tracks[tracks['set', 'subset'] <= 'small']

    trackGenres = pd.DataFrame(small['track', 'genre_top'])
    trackGenres.columns = ['Genre']
    trackGenres.reset_index(level=0, inplace=True)

    for root, dirnames, files in os.walk(aud_dir):
        if dirnames == []:
            for file in files:
                f = file[:-4]
                g = list(trackGenres[trackGenres['track_id'] == int(f)]['Genre'])
                track_ids.append((f,root,g[0]))

    return track_ids

# Input - Audio file
# Return - Shape of a melspectrogram
# To try - Find out different values of shapes, take median as the default shape
def getDefaultShape():
    tempFeatures, _ = load_track(AUDIO_DIR + "009/009152.mp3")
    return tempFeatures.shape

# Here, multiple pickles are created which puts less strain on the computer memory than a single
# large pickle data creation. We have divided into 16 pickle files, each has 500 tracks
# Input - Main directory, tracks dataset
# Return - Pickled data
def createMultiplePickles(data_dir, trackList):
    defaultShape = getDefaultShape()

    # Computationally expensive, our hardware stops working while running this part
    # X1 = np.zeros((TRACK_COUNT,) + defaultShape, dtype=np.float32)
    # Y1 = np.zeros((TRACK_COUNT, len(GENRES)), dtype=np.float32)
    # track_paths = {}
    #
    # for i, track in enumerate(trackList):
    #     path = track[1] + "/" + str(track[0]) + ".mp3"
    #     X1[i], _ = load_track()
    #     Y1[j] = genresDict[track[2]]
    #     track_paths[track[0]] = path

    trackList = np.array_split(np.array(trackList), 16)
    for i in tqdm(range(len(trackList))):
        temp = trackList[i].tolist()
        T_COUNT = len(temp)

        # np.zeros makes 500 lists which have rows and columns shaped according to defaultShape
        X = np.zeros((T_COUNT,) + defaultShape, dtype=np.float32)
        y = np.zeros((T_COUNT, len(GENRES)), dtype=np.float32)
        track_paths = {}
        # print(X.shape, y.shape)

        for j, track in enumerate(temp):
            try:
                path = track[1] + "/" + str(track[0]) + ".mp3"
#                if j % 100 == 0:
#                    print(path)
                X[j], _ = load_track(path, defaultShape)
                y[j] = genresDict[track[2]]
                track_paths[track[0]] = path
                except:
                    pass

        data = {'X': X, 'y': y, 'track_paths': track_paths}
        with open(PICKLE_DIR + "data" + str(i) + ".pkl", 'wb') as f:
            dump(data, f)

# Combining all the 16 pickle data files to one pickle. Whenever our hardware takes too much
# time to compute something, we run the same code on Google Colab.
def combinedPickle():
    pickle_files = []
    for (dirpath, dirnames, filenames) in os.walk(PICKLE_DIR):
        pickle_files.extend(filenames)

    data_dict = dict()
    data_dict['X'] = []
    data_dict['y'] = []
    data_dict['track_paths'] = []
    for pickle_file_index, pickle_file_name in enumerate(pickle_files):
        print(pickle_file_name)
        with open(PICKLE_DIR + pickle_file_name, 'rb') as pickle_file:
            pickle_data = pickle.load(pickle_file)
            data_dict['X'].extend(pickle_data['X'])
            data_dict['y'].extend(pickle_data['y'])
            data_dict['track_paths'].extend(pickle_data['track_paths'])

    data_dict['X'] = np.array(data_dict['X'])
    data_dict['y'] = np.array(data_dict['y'])
    with open(PICKLE_DIR + 'finalé.pkl', 'wb') as final_pickle:
        pickle.dump(data_dict, final_pickle)

if __name__ == "__main__":
    genres_data = pd.read_csv(METADATA_DIR + "genres.csv", index_col = 0)
    tracks = cleanTracksData(METADATA_DIR + "tracks2.csv")
    genresDict = {}

    # One hot encoding genre list, which is our output
    labelEncoded = LabelEncoder().fit_transform(GENRES)
    labelEncoded = labelEncoded.reshape(len(labelEncoded), 1)
    oneHotEncoder = OneHotEncoder(sparse=False)
    oneHotEncoded = oneHotEncoder.fit_transform(labelEncoded)

    for i, genre in enumerate(GENRES):
        genresDict[genre] = np.array(oneHotEncoded[i])

    trackIDs = getTrackIDs(AUDIO_DIR, tracks)
    # text file made to do some quality control test on excel
    np.savetxt(MAIN_DIR + "trackIDs.csv", trackIDs, delimiter=",", fmt='%s')

    if not os.path.exists(PICKLE_DIR):
        try:
            os.makedirs(PICKLE_DIR)
        except:
            pass
            
    createMultiplePickles(DATA_DIR, trackIDs)
    combinedPickle()
