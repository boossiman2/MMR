# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle
import numpy as np
import os


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load Poster feature map embedded by EfficientNetV2B2
    with open('./data/poster_feature_map.pickle', 'rb') as fr:
        feature_map = pickle.load(fr)
    # Load Plot feature embedded by Glove or Word2Vec


