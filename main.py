from fastapi import FastAPI
import tensorflow_hub as hub
import numpy as np
from pydantic import BaseModel
from pymongo import MongoClient
import openai

app = FastAPI()

openai.api_key = "sk-..."
class Texts(BaseModel):
    text1: str

# Load the USE model
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Connect to MongoDB
client = MongoClient("")
db = client["mydatabase"]
collection = db["mycollection"]

def chatgptapi(prompt):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    return completion.choices[0].message.content

def calculate_use_embeddings(text):
    embeddings = use_model([text])[0]
    embeddings_np = embeddings.numpy()
    return embeddings_np.tolist()

# Function to calculate USE similarity
def calculate_use_similarity(embeddings1, embeddings2):
    similarity = np.inner(embeddings1, embeddings2)
    return float(similarity)

# Create an endpoint for chat and saving to DB
@app.post("/chat/")
async def chat_and_save(request: Texts):

    new_text = request.text1

    # Check if the collection is empty
    if collection.count_documents({}) == 0:
        # If the collection is empty, save the new text and embeddings directly
        embeddings = calculate_use_embeddings(new_text)
        gpt_response = chatgptapi(new_text)
        data = {"text1": new_text, "embeddings": embeddings, "gpt_response": gpt_response}
        collection.insert_one(data)
        return {"message": "New text and embeddings saved because the database was empty.", "gpt_response": gpt_response}
    
    # If the collection is not empty, compare similarity with existing texts
    gpt_response = None  # Initialize GPT response as None
    new_text_embeddings = calculate_use_embeddings(new_text)

    for existing_doc in collection.find():
        existing_text = existing_doc.get("text1", "")
        existing_embeddings = existing_doc.get("embeddings", [])
        similarity_score = calculate_use_similarity(new_text_embeddings, existing_embeddings)

        if similarity_score > 0.75:
            # If similarity is above 0.75, get the GPT response from the database
            gpt_response = existing_doc.get("gpt_response")
            break

    # If gpt_response is still None, it means no similar text with similarity > 0.75 was found
    if gpt_response is None:
        gpt_response = chatgptapi(new_text)
        data = {"text1": new_text, "embeddings": new_text_embeddings, "gpt_response": gpt_response}
        collection.insert_one(data)
        return {"gpt_response": gpt_response}
    
    return {"gpt_response": gpt_response}
