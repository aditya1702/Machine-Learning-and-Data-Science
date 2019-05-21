# MusicGenre
"Music Genre Recognition throgh Audio Samples" is the final project submission for course CS543: Massive Data Storage
and Retrieval.
Maintain the following structure to implement the project from scratch. (Files to be added to data folder)

/Data

/Data/fma_metadata

/Data/fma_small

Description of the following files:

1. util.py - Loads common functions and variables
2. create_pickle_data.py - Cleans and creates the pickle files to be used for data loading (audio is converted to spectrograms and then numpy arrays)
3. train_model.py - Model architecture
4. model_to_js.py - Convert the model to json format, so that it can be used for web application
5. Go to /static in terminal and run python -m http.server 8000 (or whichever port you want). It should load the model and web on localhost. Chrome preferred.
