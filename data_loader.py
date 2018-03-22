import numpy as np
import pandas as pd


'''sample data loading methods to numpy'''


def load_linearized(path,b_train):
    ''' train.json or test.json '''
    dataset = pd.read_json(path)
    band_1 = np.concatenate([im for im in dataset['band_1']]).reshape(-1, 75*75)
    band_2 = np.concatenate([im for im in dataset['band_2']]).reshape(-1, 75* 75)
    inc_angle = dataset["inc_angle"]
    if b_train :
        labels = dataset["is_iceberg"]
        return band_1,band_2,inc_angle, labels
    else:
        return band_1,band_2,inc_angle, dataset["id"]




def load_data(path,b_train):
    train_set = pd.read_json(path)
    band_1 = np.concatenate([im for im in train_set['band_1']]).reshape(-1, 75,75)
    band_2 = np.concatenate([im for im in train_set['band_2']]).reshape(-1, 75, 75)
    data = np.stack ((band_1,band_2))
    data = np.moveaxis (data,[0,1,2,3],[3,0,1,2])
    print(data.shape)
    if b_train :
        labels = train_set["is_iceberg"]
        return data, labels
    else :
        return data


def load_data_2d(path, b_train):
        dataset = pd.read_json(path)
        band_1 = np.concatenate([im for im in dataset['band_1']]).reshape(-1, 75, 75)
        band_2 = np.concatenate([im for im in dataset['band_2']]).reshape(-1, 75, 75)
        inc_angle = dataset["inc_angle"]
        if b_train:
            labels = dataset["is_iceberg"]
            return band_1,band_2, inc_angle, labels
        else:
            return band_1,band_2, inc_angle,dataset["id"]


