import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, metrics
from func import scaling, split_data, binatodeci
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures, MultiLabelBinarizer
from sklearn.metrics import f1_score

# get data from files


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

file = pd.read_csv(r'..\Dataset\tmdb-movies.csv')
file.drop(columns=(['id','imdb_id','tagline','release_year']), inplace=True)
file.drop(columns=(['overview','homepage','original_title','keywords']), inplace=True)
file = file[['cast','genres','production_companies','director','release_date','popularity','runtime','vote_count','vote_average','budget_adj','revenue_adj']] # arrange columns

# cast - keywords - genres - production_companies there is a null value 
file.dropna(inplace=True)
file = file[file['budget_adj'] != 0].dropna()
file = file[file['revenue_adj'] != 0].dropna()
file.drop_duplicates(inplace=True)
file.reset_index(drop=True,inplace=True)


# Scaling by built in function or create a new one
file['release_date'] = pd.to_datetime(file['release_date'])
file['release_date'] = file['release_date'].apply(pd.Timestamp.timestamp)
scaling(file,['popularity','runtime','vote_count','vote_average','budget_adj','revenue_adj','release_date'])

split_data(file,['cast','genres','production_companies'])


label_encoder_cast = LabelEncoder()
cast_data = file['cast'].explode()
cast_data[:] = label_encoder_cast.fit_transform(cast_data)
file['cast'] = cast_data.groupby(level=0).agg(list)

label_encoder_genres = LabelEncoder()
genres_data = file['genres'].explode()
genres_data[:] = label_encoder_genres.fit_transform(genres_data)
file['genres'] = genres_data.groupby(level=0).agg(list)

label_encoder_production_companies= LabelEncoder()
production_companies_data = file['production_companies'].explode()
production_companies_data[:] = label_encoder_production_companies.fit_transform(production_companies_data)
file['production_companies'] = production_companies_data.groupby(level=0).agg(list)

label_encoder_director = LabelEncoder()
director_data = file['production_companies'].explode()
director_data[:] = label_encoder_director.fit_transform(director_data)
file['director'] = director_data.groupby(level=0).agg(list)


file = file.explode('cast')
file = file.explode('genres')
file = file.explode('production_companies')
file = file.explode('director')
file.reset_index(drop=True,inplace = True)
file.cast = file.cast.astype('int') 

file.genres = file.genres.astype('int') 
file.production_companies = file.production_companies.astype('int') 
file.director = file.director.astype('int') 
scaling(file,['cast','production_companies','genres','director'])
X = file.iloc[:,0:9]
Y = file['budget_adj'] - file['revenue_adj']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

poly_features = PolynomialFeatures(degree= 3)

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

print(poly_model.score(poly_features.fit_transform(X_test),y_test))


