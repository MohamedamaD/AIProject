import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from func import scaling, split_data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# get data from files
file = pd.read_csv(r'..\Dataset\tmdb-movies.csv')

# clean & filter
'''
-------- -------- -------- -------- --------
    1st filter & drop columns 
    2nd fill columns which have a nan values 
    3th remove duplicate records
    4th normalization (we using scaling because there is a variety of magnitude of data)
    5th convert categorical to numerical
-------- -------- -------- -------- --------
'''
file.drop(columns=(['id','imdb_id','tagline','release_year']), inplace=True)
file.drop(columns=(['director','overview','homepage','original_title']), inplace=True)
file = file[['cast','keywords','genres','production_companies','release_date','popularity','runtime','vote_count','vote_average','budget_adj','revenue_adj']] # arrange columns

# cast - keywords - genres - production_companies there is a null value 
file.dropna(inplace=True)
file.reset_index(drop=True,inplace=True)

file.drop_duplicates(inplace=True)

# Scaling by built in function or create a new one
scaling(file,['popularity','runtime','vote_count','vote_average','budget_adj','revenue_adj'])
'''
    min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1))
    file['popularity'] = min_max_scaler.fit_transform(file.iloc[:,0:1])
    data_after_scaling = min_max_scaler.fit_transform(file.iloc[:,7:])
    df = pd.DataFrame(data_after_scaling)
    file['vote_count'] = df[0]
    file['vote_average'] = df[1]
    file['budget_adj'] = df[2]
    file['revenue_adj'] = df[3]
    data_after_scaling = min_max_scaler.fit_transform(file.iloc[:,3:4])
    file['runtime'] = data_after_scaling
    del data_after_scaling, df, min_max_scaler
'''

split_data(file,['cast','keywords','genres','production_companies'])

label_encoder_cast = LabelEncoder()
cast_data = file['cast'].explode()
cast_data[:] = label_encoder_cast.fit_transform(cast_data)
file['cast'] = cast_data.groupby(level=0).agg(list)

label_encoder_keywords = LabelEncoder()
keywords_data = file['keywords'].explode()
keywords_data[:] = label_encoder_keywords.fit_transform(keywords_data)
file['keywords'] = keywords_data.groupby(level=0).agg(list)

label_encoder_genres = LabelEncoder()
genres_data = file['genres'].explode()
genres_data[:] = label_encoder_genres.fit_transform(genres_data)
file['genres'] = genres_data.groupby(level=0).agg(list)

label_encoder_production_companies= LabelEncoder()
production_companies_data = file['production_companies'].explode()
production_companies_data[:] = label_encoder_production_companies.fit_transform(production_companies_data)
file['production_companies'] = production_companies_data.groupby(level=0).agg(list)