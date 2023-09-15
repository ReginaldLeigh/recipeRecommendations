from flask import Flask, render_template, request, flash, Response, url_for
import json
import pickle
from recipeList import get_recipes
from machineLearning import machineLearning

app = Flask(__name__)
app.secret_key="secret"


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        flash("Welcome to Tasty Foods")
        flash("Begin typing to select your favorite recipe and receive recommendations")
    return render_template("index.html")

@app.route("/_autocomplete", methods=['GET'])
def autocomplete():
    recipes = get_recipes()
    return Response(json.dumps(recipes), mimetype='application/json')

@app.route("/search", methods=['POST', 'GET'])
def search():
    resultsHeader, results = machineLearning(request.form['user_input'])
    return render_template("index.html", resultsHeader=resultsHeader, results=results)

if __name__ == '__main__':
    app.run(debug=True)





