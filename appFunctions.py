import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from zipfile import ZipFile
import pickle
import io
import boto3

def get_datasets(access_key, secret_key):
    s3 = boto3.resource('s3', 
    endpoint_url = 'https://88e4bfb043c52100374debae54b3ea31.r2.cloudflarestorage.com',
    aws_access_key_id = access_key, 
    aws_secret_access_key = secret_key
    )

    bucket = s3.Bucket('recipe-app-files')
    file_list = []

    for item in bucket.objects.all():
        body = item.get()['Body'].read()
        file_list.append(body)
    
    pickles = []

    for file in file_list:
      zf = ZipFile(io.BytesIO(file))
      pkl = pickle.load(zf.open(zf.namelist()[0]))
      zf.close()
      pickles.append(pkl)

    return pickles

def recipe_recommendations(recipe_name, clean_data, feature_matrix):
    recipe_index = clean_data[clean_data['name'] == recipe_name].index
    recipe_index = recipe_index[0]
    recipe_id = clean_data['recipe_id'].loc[recipe_index]
    recipe_data = pd.DataFrame(feature_matrix.loc[[recipe_index]])
    cosine_sim = cosine_similarity(recipe_data, feature_matrix)
    cosine_scores = pd.DataFrame(cosine_sim, index=[recipe_id], columns=clean_data['recipe_id']).T.drop(recipe_id)
    recommended_ids = cosine_scores.nlargest(10, recipe_id).index
    recommended_recipes = pd.DataFrame()

    for id in recommended_ids:
        recipe = clean_data[clean_data['recipe_id'] == id]
        recommended_recipes = pd.concat([recommended_recipes, recipe], ignore_index=True)

    return recommended_recipes.to_dict(orient='records')

def get_all_recipes(recipe_data):
    recipes = recipe_data['name']
    return recipes.tolist()



def get_tag_data(feature_matrix):
    tag_totals = feature_matrix.sum()[:-2]
    tags_df = pd.DataFrame({'tag': tag_totals.index, 'total_count': tag_totals.values}).sort_values(by='total_count')
    tags_df = tags_df.nlargest(10, 'total_count', keep='first')
    tagchart_data = tags_df.values.tolist()
    tagchart_labels = [row[0] for row in tagchart_data]
    tagchart_values = [row[1] for row in tagchart_data]
    return tagchart_labels, tagchart_values

def get_rating_data(results):
    rating_df = pd.DataFrame(columns=['recipe_id', 'avgRating', 'numRatings'])
    
    for row in range(len(results)):
        df = pd.DataFrame({'recipe_id': [results[row]['recipe_id']],
                       'avgRating': [results[row]['avgRating']],
                       'numRatings': [results[row]['numRatings']]})
        
        rating_df = pd.concat([rating_df, df], ignore_index=True)

    ratingchart_data = rating_df.values.tolist()
    ratingchart_labels = [row[0] for row in ratingchart_data]
    ratingchart_avgRatings = [row[1] for row in ratingchart_data]
    ratingchart_numRatings = [row[2] for row in ratingchart_data]
    return ratingchart_labels, ratingchart_avgRatings, ratingchart_numRatings

def get_nutrition_data(results):
    # Redundant code is to properly set data types....and I'm tired
    nutrition_df = pd.DataFrame({
            'recipe_id': [results[0]['recipe_id']],
            'total fat (PDV)': [results[0]['total fat (PDV)']],
            'sugar (PDV)': [results[0]['sugar (PDV)']],
            'sodium (PDV)': [results[0]['sodium (PDV)']],
            'protein (PDV)': [results[0]['protein (PDV)']],
            'saturated fat (PDV)': [results[0]['saturated fat (PDV)']],
            'carbohydrates (PDV)': [results[0]['carbohydrates (PDV)']]
            })
    
    for row in range(1, 3):
        df = pd.DataFrame({
            'recipe_id': [results[row]['recipe_id']],
            'total fat (PDV)': [results[row]['total fat (PDV)']],
            'sugar (PDV)': [results[row]['sugar (PDV)']],
            'sodium (PDV)': [results[row]['sodium (PDV)']],
            'protein (PDV)': [results[row]['protein (PDV)']],
            'saturated fat (PDV)': [results[row]['saturated fat (PDV)']],
            'carbohydrates (PDV)': [results[row]['carbohydrates (PDV)']]
            })
        
        nutrition_df = pd.concat([nutrition_df, df], ignore_index=True)

    nutri_chart_data = nutrition_df.values.tolist()
    nutri_chart_ids = [row[0] for row in nutri_chart_data]
    nutri_chart_labels = ['total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']
    nutri_data1 = nutri_chart_data[0][1:(len(nutrition_df.columns))]
    nutri_data2 = nutri_chart_data[1][1:(len(nutrition_df.columns))]
    nutri_data3 = nutri_chart_data[2][1:(len(nutrition_df.columns))]
    nutri_data = [nutri_data1, nutri_data2, nutri_data3]
    return nutri_chart_ids, nutri_chart_labels, nutri_data


def get_random_recipe(recipe_data):
    size = len(recipe_data) - 1
    random_idx = np.random.randint(size)
    name = recipe_data['name'].loc[random_idx]
    return name
    


