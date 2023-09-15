import csv
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


df_recipes = pd.read_csv("./fooddata/RAW_recipes-short.csv", usecols=['id', 'name'])[['id', 'name']]
df_recipes.rename(columns={'id': 'recipe_id'}, inplace=True)

df_users = pd.read_csv("./fooddata/RAW_interactions-short.csv", usecols=['user_id', 'recipe_id', 'rating'])
df_interactions = pd.merge(df_users, df_recipes, on='recipe_id')

combined_rating = df_interactions.dropna(axis=0, subset=['name'])
rating_count = (combined_rating.groupby(by = ['name'])['rating']).count().reset_index()
rating_count.rename(columns={'rating': 'totalRatingCount'}, inplace=True)

df_aggData = df_interactions.merge(rating_count, left_on='name', right_on='name', how='left')
popularity_threshold = 4
df_popularRecipes = df_aggData.query('totalRatingCount >= @popularity_threshold')
df_popularRecipes = df_popularRecipes.drop_duplicates(subset=['name'])
recipes = df_popularRecipes['name']
recipelist = recipes.tolist()

def get_recipes():
    return recipelist




    


