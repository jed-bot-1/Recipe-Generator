from fastapi import FastAPI
from pydantic import BaseModel
from tokenizer_utils import comma_tokenizer
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

app = FastAPI()

df = joblib.load("recipe_data.pkl")
vectorizer = joblib.load("vectorizer.pkl")
ingredient_matrix = joblib.load("ingredient_matrix.pkl")

@app.get("/")
async def root():
    return {"message":"The Server is running!!"}

#making sure it accept list of string because that is the output of the image recogntion Server
class InputIngredients(BaseModel):
    ingredients: list[str]


@app.post("/recommend")
def recommend(ingredients:InputIngredients):
    normalized = [ingredient.lower() for ingredient in ingredients.ingredients]
    input = " ".join(normalized)
    input_vec = vectorizer.transform([input])
    similarities = cosine_similarity(input_vec, ingredient_matrix).flatten()
    top_indices = similarities.argsort()[::-1]

    recommendations = []
    for i in top_indices:
        recipe = df.iloc[i].to_dict()
        recipe['similarity'] = float(similarities[i])
        recommendations.append(recipe)

    return{"recommendations":recommendations}