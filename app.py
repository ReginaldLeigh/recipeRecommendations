from flask import Flask, render_template, request, flash, Response, url_for
import json
import pickle
import pandas as pd
import numpy as np
from appFunctions import *
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
app.secret_key=os.getenv('FLASK_SECRET')


clean_data = get_datasets(os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY'))[0]
feature_matrix = get_datasets(os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY'))[1]


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        flash("Welcome to Tasty Foods")
        flash("Begin typing to select your favorite recipe and receive recommendations")
    return render_template("index.html", visibility="hidden")

@app.route("/_autocomplete", methods=['GET'])
def autocomplete():
    recipes = get_all_recipes(clean_data)
    return Response(json.dumps(recipes), mimetype='application/json')

@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    results = recipe_recommendations(request.form['user_input'], clean_data, feature_matrix)
    tag_labels, tag_values = get_tag_data(feature_matrix)
    rating_labels, avgRating_values, numRating_values = get_rating_data(results)
    nutri_ids, nutri_labels, nutri_data = get_nutrition_data(results)
    return render_template("index.html", results=results, tag_labels=tag_labels, tag_values=tag_values, 
                           rating_labels=rating_labels, avgRating_values=avgRating_values, numRating_values=numRating_values,
                           nutri_ids=nutri_ids, nutri_labels=nutri_labels, nutri_data=nutri_data,
                           visibility="visible")

@app.route("/_randomize", methods=['GET'])
def randomize():
    name = get_random_recipe(clean_data)
    return Response(name)


if __name__ == '__main__':
    app.run(debug=True)



