# %%
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pickle

def machineLearning(user_input):
    df_recipes = pd.read_csv("./fooddata/RAW_recipes-short.csv", usecols=['id', 'name'])[['id', 'name']]
    df_recipes.rename(columns={'id': 'recipe_id'}, inplace=True)
    
    df_users = pd.read_csv("./fooddata/RAW_interactions-short.csv", usecols=['user_id', 'recipe_id', 'rating'])
    df_interactions = pd.merge(df_users, df_recipes, on='recipe_id')

    combined_rating = df_interactions.dropna(axis=0, subset=['name'])
    rating_count = (combined_rating.groupby(by = ['name'])['rating']).count().reset_index()
    rating_count.rename(columns={'rating': 'totalRatingCount'}, inplace=True)

    df_aggData = df_interactions.merge(rating_count, left_on='name', right_on='name', how='left')

    # pd.set_option('display.float_format', lambda x: '%.3f' % x)
    # print(df_aggData['totalRatingCount'].describe())

    # Sets threshold for which to recommend a recipe
    popularity_threshold = 4
    df_popularRecipes = df_aggData.query('totalRatingCount >= @popularity_threshold')

    # Creates feature pivot table for each recipe
    df_recipeFeatures = df_popularRecipes.pivot_table(index='recipe_id', columns='user_id', values='rating').fillna(0)

    # Creates feature matrix for each recipe (cosine similarity)
    df_recipeMatrix = csr_matrix(df_recipeFeatures.values)

    # Create K-NearestNeighbor model
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
    knn_model.fit(df_recipeMatrix)

  
    # Select recipe from matrix using index
    recipe_id = df_recipes.query('name == @user_input').iloc[0]['recipe_id']
    for i in range(df_recipeFeatures.shape[0]):
        if df_recipeFeatures.index[i] == recipe_id:
            query_index = i

    # Create list of recommendations based off recipes with similar feature set (Collabrative Filtering)
    distances, indices = knn_model.kneighbors(df_recipeFeatures.iloc[query_index,:].values.reshape(1,-1), n_neighbors=11)
    results = []

    for i in range(0, len(distances.flatten())):
        if i == 0:                                              # First item is always itself
            id = df_recipeFeatures.index[query_index]
            recipe_name = df_recipes.query('recipe_id == @id').iloc[0]['name']
            resultsHeader = 'Top 10 recommendations for "{0}":\n'.format(recipe_name)
        else:
            id = df_recipeFeatures.index[indices.flatten()[i]]
            recipe_name = df_recipes.query('recipe_id == @id').iloc[0]['name']
            results.append('{0}: {1}, with distance of {2}'.format(i, recipe_name, distances.flatten()[i]))


    return resultsHeader, results
    

