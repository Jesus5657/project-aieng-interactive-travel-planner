import os
import json
import requests
import faiss
import numpy as np
import openai
import streamlit as st
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Configure API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "OPENWEATHER")

openai.api_key = OPENAI_API_KEY
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load data from the local folder
def load_text_data(folder_path):
    structured_data = {}
    if not os.path.exists(folder_path):
        return {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                structured_data[filename.replace(".txt", "")] = file.read().strip()
    return structured_data

data_paths = {
    "landmarks": r"C:\Users\jesus\project-aieng-interactive-travel-planner\data\landmarks",
    "municipalities": r"C:\Users\jesus\project-aieng-interactive-travel-planner\data\municipalities",
    "news": r"C:\Users\jesus\project-aieng-interactive-travel-planner\data\elmundo_chunked_es_page1_40years"
}

landmarks = load_text_data(data_paths["landmarks"])
municipalities = load_text_data(data_paths["municipalities"])
news_articles = load_text_data(data_paths["news"])

# Create FAISS index
embedding_dim = 384  
index = faiss.IndexFlatL2(embedding_dim)
descriptions = list(landmarks.values()) + list(municipalities.values())

if descriptions:
    embeddings = np.array([model.encode(desc) for desc in descriptions], dtype=np.float32)
    
    if embeddings.size > 0:  # Ensure embeddings exist before adding
        index.add(embeddings)
        st.write(f"Embeddings added to FAISS: {index.ntotal}")
    else:
        st.write("⚠️ No valid embeddings were generated.")

location_keys = list(landmarks.keys()) + list(municipalities.keys())

# Function to get weather forecast
def find_weather_forecast(date, location):
    try:
        lat, lon = (18.4655, -66.1057)  # Default coordinates for Puerto Rico
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url).json()
        
        if "list" in response and len(response["list"]) > 0:
            forecast = response["list"][0]
            weather_description = forecast["weather"][0]["description"].capitalize()
            temperature = forecast["main"]["temp"]
            return f"{weather_description}, {temperature}°C"
        else:
            return "Weather data not available."
    except Exception as e:
        return f"Error retrieving weather data: {e}"

# Function to find locations based on user interests
def rank_appropriate_locations(user_prompt):
    if index.ntotal == 0:
        st.warning("⚠️ No data available. Make sure landmarks and municipalities are loaded.")
        return []
    
    query_vector = model.encode(user_prompt).astype("float32").reshape(1, -1)
    _, ranked_indices = index.search(query_vector, k=min(10, index.ntotal))
    
    if len(ranked_indices) == 0 or len(ranked_indices[0]) == 0:
        return ["No relevant locations found"]
    
    return [location_keys[i] for i in ranked_indices[0] if i < len(location_keys)]

# Function to retrieve the municipality of a location
def get_municipality(location):
    for municipality in municipalities:
        if location.lower() in municipalities[municipality].lower():
            return municipality
    return "Unknown Municipality"

# Streamlit User Interface
st.title("🌴 Puerto Rico Travel Planner")
st.sidebar.header("Travel Options")

# Define a visit list in session state
if "visit_list" not in st.session_state:
    st.session_state.visit_list = []

# User input
travel_date = st.sidebar.date_input("📅 When are you planning to travel?", datetime.today())
user_interests = st.text_input("📝 What kind of places would you like to visit?", "")

if user_interests:
    suggested_locations = rank_appropriate_locations(user_interests)
    
    if "No relevant locations found" in suggested_locations:
        st.warning("No locations found for your interests. Try different keywords.")
    else:
        st.subheader("🏝️ Recommended Places for You")
        
        for i, loc in enumerate(suggested_locations, 1):
            municipality = get_municipality(loc)
            st.write(f"**{i}. {loc.replace('_', ' ').title()}** - {municipality}")
            
            if st.button(f"Add {loc}", key=f"add_{i}"):
                st.session_state.visit_list.append(loc)
                st.success(f"{loc.replace('_', ' ').title()} added to your visit list!")

st.subheader("🗺️ Your Visit List")
if st.session_state.visit_list:
    for loc in st.session_state.visit_list:
        municipality = get_municipality(loc)
        weather_info = find_weather_forecast(travel_date, loc)
        st.write(f"✅ **{loc.replace('_', ' ').title()}** ({municipality}) - {weather_info}")
    
    if st.button("Finalize Trip Plan"):
        st.success("Your trip plan has been finalized! Check the Finalized Travel Plan section.")
        st.session_state.finalized = True
else:
    st.write("You haven't added any places to your list yet.")

st.sidebar.subheader("📌 Finalize Plan")
if st.sidebar.button("View Final Plan"):
    st.subheader("📍 Finalized Travel Plan")
    for loc in st.session_state.visit_list:
        municipality = get_municipality(loc)
        st.write(f"📍 **{loc.replace('_', ' ').title()}** - {municipality}")

st.sidebar.subheader("🔄 Reset")
if st.sidebar.button("Reset List"):
    st.session_state.visit_list = []
    st.experimental_rerun()
