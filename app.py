import types
import sys
import joblib
import pandas as pd
import numpy as np
import gc
import logging
import os
import psutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey-patch identity_tokenizer for vectorizer
def identity_tokenizer(text):
    return text

fake_main = types.ModuleType("__main__")
fake_main.identity_tokenizer = identity_tokenizer
sys.modules["__main__"] = fake_main

# Initialize FastAPI
app = FastAPI(
    title="Bicolano Recipe Recommender",
    description="API for recommending Bicolano recipes based on available ingredients",
    version="1.0.0"
)

# Global models and data
vectorizer = None
nn_model = None
df = None

@app.on_event("startup")
async def load_model():
    global vectorizer, nn_model, df
    try:
        logger.info("📦 Loading models and recipe data...")
        vectorizer = joblib.load("vectorizer.joblib")
        nn_model = joblib.load("recipe_recommender_model.joblib")
        df = pd.read_csv("cleaned_bicolano_recipes.csv")
        logger.info(f"✅ Loaded {len(df)} recipes.")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise e

# Pydantic models
class IngredientsRequest(BaseModel):
    ingredients: List[str] = Field(
        ..., description="List of available ingredients",
        example=["coconut milk", "pork", "chili", "onion"]
    )
    max_recipes: Optional[int] = Field(
        default=5,
        description="Maximum number of recipe recommendations to return",
        ge=1,
        le=20
    )

class Recipe(BaseModel):
    dish_name: str
    ingredients: Optional[str] = None
    instructions: Optional[str] = None
    similarity_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    input_ingredients: List[str]
    recommended_recipes: List[Recipe]
    total_found: int
    message: str

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_recipes(data: IngredientsRequest):
    try:
        if vectorizer is None or nn_model is None or df is None:
            raise HTTPException(status_code=500, detail="Models not loaded properly")

        if not data.ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")

        # Clean input
        user_ingredients = [i.strip().lower() for i in data.ingredients if i.strip()]
        if not user_ingredients:
            raise HTTPException(status_code=400, detail="No valid ingredients provided")

        logger.info(f"👨‍🍳 User ingredients: {user_ingredients}")

        ingredient_names = vectorizer.get_feature_names_out()
        user_vector = np.zeros(len(ingredient_names))
        matched_ingredients = []

        for ing in user_ingredients:
            if ing in ingredient_names:
                idx = list(ingredient_names).index(ing)
                user_vector[idx] = 1
                matched_ingredients.append(ing)

        user_vector = user_vector.reshape(1, -1)
        logger.info(f"✅ Matched ingredients: {matched_ingredients}")

        distances, indices = nn_model.kneighbors(
            user_vector,
            n_neighbors=min(data.max_recipes, len(df))
        )

        recommended_recipes = []
        for i, idx in enumerate(indices.flatten()):
            row = df.iloc[idx]
            score = max(0, 1 - distances[0][i])

            recipe = Recipe(
                dish_name=str(row['recipe_name']),
                ingredients=row.get('ingredients', None),
                instructions=row.get('instructions', None),
                similarity_score=round(score, 3)
            )
            recommended_recipes.append(recipe)

        response = RecommendationResponse(
            input_ingredients=user_ingredients,
            recommended_recipes=recommended_recipes,
            total_found=len(recommended_recipes),
            message=f"Found {len(recommended_recipes)} recipe recommendations based on your ingredients"
        )

        logger.info("🎉 Recommendation complete.")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # 🧹 Memory cleanup
        try:
            del user_ingredients, user_vector, matched_ingredients
            del distances, indices, row, recipe, recommended_recipes
        except Exception as cleanup_error:
            logger.warning(f"[Cleanup] Delete failed: {cleanup_error}")
        
        gc.collect()

        try:
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / (1024 * 1024)
            logger.info(f"📉 Memory after cleanup: {mem:.2f} MB")
        except Exception as mem_error:
            logger.warning(f"[Memory] Check failed: {mem_error}")

@app.get("/")
async def root():
    return {
        "message": "Bicolano Recipe Recommender API",
        "status": "active",
        "total_recipes": len(df) if df is not None else 0
    }

@app.get("/debug/columns")
async def get_columns():
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    return {
        "columns": df.columns.tolist(),
        "sample_row": df.iloc[0].to_dict() if len(df) > 0 else {},
        "total_rows": len(df)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if all([vectorizer, nn_model, df is not None]) else "unhealthy",
        "models_loaded": {
            "vectorizer": vectorizer is not None,
            "nn_model": nn_model is not None,
            "recipes_data": df is not None
        },
        "total_recipes": len(df) if df is not None else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
