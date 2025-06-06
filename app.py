import types
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkey-patch identity_tokenizer so joblib load works for vectorizer
def identity_tokenizer(text):
    return text

fake_main = types.ModuleType("__main__")
fake_main.identity_tokenizer = identity_tokenizer
sys.modules["__main__"] = fake_main

app = FastAPI(
    title="Bicolano Recipe Recommender",
    description="API for recommending Bicolano recipes based on available ingredients",
    version="1.0.0"
)

# Global variables for models and data
vectorizer = None
nn_model = None
df = None

@app.on_event("startup")
async def load_model():
    """Load the ML models and recipe data on startup"""
    global vectorizer, nn_model, df
    try:
        logger.info("Loading models and data...")
        vectorizer = joblib.load("vectorizer.joblib")
        nn_model = joblib.load("recipe_recommender_model.joblib")
        df = pd.read_csv("cleaned_bicolano_recipes.csv")
        logger.info(f"Successfully loaded {len(df)} recipes")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise e

class IngredientsRequest(BaseModel):
    """Request model for ingredient input"""
    ingredients: List[str] = Field(
        ..., 
        description="List of available ingredients", 
        example=["coconut milk", "pork", "chili", "onion"]
    )
    max_recipes: Optional[int] = Field(
        default=5, 
        description="Maximum number of recipe recommendations to return",
        ge=1,
        le=20
    )

class Recipe(BaseModel):
    """Recipe model for structured output"""
    dish_name: str
    ingredients: Optional[str] = None
    instructions: Optional[str] = None
    similarity_score: Optional[float] = None

class RecommendationResponse(BaseModel):
    """Response model for recipe recommendations"""
    input_ingredients: List[str]
    recommended_recipes: List[Recipe]
    total_found: int
    message: str

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_recipes(data: IngredientsRequest):
    """
    Recommend Bicolano recipes based on available ingredients
    """
    try:
        # Validate that models are loaded
        if vectorizer is None or nn_model is None or df is None:
            raise HTTPException(status_code=500, detail="Models not loaded properly")
        
        # Clean and validate input ingredients
        if not data.ingredients:
            raise HTTPException(status_code=400, detail="No ingredients provided")
        
        # Clean ingredient names
        user_ingredients = []
        for ing in data.ingredients:
            cleaned_ing = ing.strip().lower()
            if cleaned_ing:  # Only add non-empty ingredients
                user_ingredients.append(cleaned_ing)
        
        if not user_ingredients:
            raise HTTPException(status_code=400, detail="No valid ingredients provided")
        
        logger.info(f"Processing ingredients: {user_ingredients}")
        
        # Get feature names from vectorizer
        ingredient_names = vectorizer.get_feature_names_out()
        
        # Encode ingredients using binary approach (matching your test script)
        user_vector = np.zeros(len(ingredient_names))
        matched_ingredients = []
        
        for ing in user_ingredients:
            if ing in ingredient_names:
                idx = list(ingredient_names).index(ing)
                user_vector[idx] = 1
                matched_ingredients.append(ing)
        
        user_vector = user_vector.reshape(1, -1)
        logger.info(f"Matched ingredients: {matched_ingredients}")
        
        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(
            user_vector, 
            n_neighbors=min(data.max_recipes, len(df))
        )
        
        # Build recipe recommendations with detailed information
        recommended_recipes = []
        
        # Check what columns are available in the dataframe
        available_columns = df.columns.tolist()
        logger.info(f"Available columns: {available_columns}")
        
        for i, idx in enumerate(indices.flatten()):
            recipe_row = df.iloc[idx]
            
            # Calculate similarity score (convert distance to similarity)
            similarity_score = max(0, 1 - distances[0][i])
            
            recipe = Recipe(
                dish_name=str(recipe_row['recipe_name']),  # Using recipe_name column
                ingredients=recipe_row.get('ingredients', None),
                instructions=recipe_row.get('instructions', None),
                similarity_score=round(similarity_score, 3)
            )
            recommended_recipes.append(recipe)
        
        # Create response
        response = RecommendationResponse(
            input_ingredients=user_ingredients,
            recommended_recipes=recommended_recipes,
            total_found=len(recommended_recipes),
            message=f"Found {len(recommended_recipes)} recipe recommendations based on your ingredients"
        )
        
        logger.info(f"Successfully returned {len(recommended_recipes)} recommendations")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Bicolano Recipe Recommender API",
        "status": "active",
        "total_recipes": len(df) if df is not None else 0
    }

@app.get("/debug/columns")
async def get_columns():
    """Debug endpoint to check CSV columns"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return {
        "columns": df.columns.tolist(),
        "sample_row": df.iloc[0].to_dict() if len(df) > 0 else {},
        "total_rows": len(df)
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
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