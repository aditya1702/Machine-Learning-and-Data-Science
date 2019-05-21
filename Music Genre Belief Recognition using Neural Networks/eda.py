#%%
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

sns.set()

DATA_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Data/fma_metadata/"
IMG_DIR = "/home/vedantc6/Desktop/Projects/CS543-MusicGenreRecognition/Images/"

#%%
genres_df = pd.read_csv(DATA_DIR + "genres.csv", encoding="latin-1") 
print(genres_df.head())

#%%
tracks_df = pd.read_csv(DATA_DIR + "tracks.csv", encoding="Latin-1")

# Cleaning data
idx = tracks_df.columns.get_loc("track_title")
tracks_df = tracks_df.iloc[:, :53]
print(tracks_df.head())

#%%
echonest_df = pd.read_csv(DATA_DIR + "echonest.csv", encoding="Latin-1", index_col=0, header=[0,1,2])
print(echonest_df.head())

#%%
artist_df = pd.read_csv(DATA_DIR + "raw_artists.csv", encoding="Latin-1")
print(artist_df.head())

#%%
album_df = pd.read_csv(DATA_DIR + "raw_albums.csv", encoding="Latin-1")
print(album_df.head())

#%%
features_df = pd.read_csv(DATA_DIR + "features.csv", encoding="Latin-1", index_col=0, header=[0,1,2])
print(features_df.head())

#%%
#################################
############# EDA ###############
#################################

# Number of unique number of tracks, artists, albums, genres
print('{} tracks, {} artists, {} albums, {} genres'.format(len(tracks_df), 
      len(artist_df['artist_id'].unique()), len(album_df), sum(genres_df['#tracks'] > 0)))
         
#%%
# Missing values in datasets
def num_missing(x):
    return sum(x.isnull())
    i = sum(x.isnull())
    return (i/len(x))*100

print("Missing values per column: ")

print("Tracks dataset: ")
missing_track_info = tracks_df.apply(num_missing, axis = 0) 
print(missing_track_info)
print("\nGenres dataset: " )
missing_genre_info = genres_df.apply(num_missing, axis = 0)
print(missing_genre_info)
print("\nEchonest dataset: " )
missing_echo_info = echonest_df.apply(num_missing, axis = 0)
print(missing_echo_info)
print("\nFeatures dataset: " )
missing_features_info = features_df.apply(num_missing, axis = 0)
print(missing_features_info)
missing_album_info = album_df.apply(num_missing, axis = 0)
print(missing_album_info)
missing_artist_info = artist_df.apply(num_missing, axis = 0)
print(missing_artist_info)

#%%
def missing_values_plotter(x, fileName):
    track_cols = x.columns
    percent_missing_tracks = x.isnull().sum()*100/len(x)
    missing_value_df = pd.DataFrame({'ColName': track_cols, 
                                     'percent_missing': percent_missing_tracks})
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 17)
    g = sns.barplot(missing_value_df[missing_value_df['percent_missing'] > 0]['ColName'], 
                missing_value_df[missing_value_df['percent_missing'] > 0]['percent_missing'],
                ax=ax)

    for item in g.get_xticklabels():
        item.set_rotation(90)
    
    g.set_title(fileName.split(".")[0])
    g.figure.savefig(IMG_DIR + fileName)
    
    # Columns to keep in datasets
    to_keep = list(missing_value_df[missing_value_df['percent_missing'] < 30]['ColName'])
    return to_keep

tracks_df = tracks_df[missing_values_plotter(tracks_df, "Missing_Values_Tracks_Data.png")]
album_df = album_df[missing_values_plotter(album_df, "Missing_Values_Albums_Data.png")]
artist_df = artist_df[missing_values_plotter(artist_df, "Missing_Values_Artists_Data.png")]
#%%
# Count of albums vs release years
album_df["Year"] = album_df['album_date_created'].str.split("/").str[2].str.split(" ").str[0]
temp_df = album_df[['album_id', 'Year']].groupby(['Year']).agg(['count']).reset_index()
temp_df.columns = ["Year", "#Albums"]
sns.set_style("white")

fig = plt.figure()
plt.plot(temp_df['Year'], temp_df['#Albums'])
plt.title("Number of albums through the years captured in data")
plt.xlabel('Year')
plt.ylabel('#Albums')
fig.savefig(IMG_DIR + "numAlbumsPerYear.png")

del temp_df
#%%
test = pd.DataFrame(tracks_df.track_duration.dropna())

for i, val in enumerate(test.track_duration.values):
    if "." in str(val):
        test.track_duration.values[i] = int(val)
    if "-" in str(val):
        test.track_duration.values[i] = 0

test = test.astype(int)
test = test[(test.track_duration > 0) & (test.track_duration < 1000)]
test['track_duration'] = test.track_duration
print("Mean duration of tracks present in the dataset: {} seconds".format(test.track_duration.mean()))

g = sns.countplot(test.track_duration).set_title("Counts of tracks plotted with their duration")
g.figure.savefig(IMG_DIR + "trackDuration.png")
#%%
