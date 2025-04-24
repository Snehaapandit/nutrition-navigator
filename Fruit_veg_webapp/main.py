import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

# --- Page Config ---
st.set_page_config(page_title="Calorie Estimator", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body {
        font-family: 'Inter', sans-serif;
        color: #fff;
    }

    .stApp {
        background: url("https://wallpapers.com/images/hd/fruits-and-vegetables-5462-x-3072-background-823rjsovq6rbwwho.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 16px;
        margin-top: 1rem;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #00ADB5;
    }

    .stButton>button {
        background-color: #00ADB5;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
    }

    .stButton>button:hover {
        background-color: #007E85;
    }

    footer {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# --- Calorie Dictionary ---
calorie_data = {
    "apple": 52, "banana": 96, "carrot": 41, "tomato": 18, "potato": 77,
    "capsicum": 20, "orange": 47, "cucumber": 16, "grapes": 69, "pineapple": 50,
    "watermelon": 30, "mango": 60, "onion": 40, "radish": 16, "beetroot": 43, "papaya": 43
}

# --- Nutritional Data ---
nutrition_data = pd.DataFrame({
    "Item": list(calorie_data.keys()),
    "Calories": list(calorie_data.values()),
    "Carbs (g)": [14, 27, 10, 4, 17, 4.7, 12, 3.6, 18, 13, 8, 15, 9, 3.4, 10, 11],
    "Protein (g)": [0.3, 1.3, 0.9, 0.9, 2, 0.9, 0.9, 0.7, 0.7, 0.5, 0.6, 0.8, 1.1, 0.7, 1.6, 0.5],
    "Fat (g)": [0.2, 0.3, 0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.4, 0.1, 0.1, 0.2, 0.3]
})

# --- Load Model Once ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "trained_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# --- Sidebar with Logo ---
st.sidebar.image("https://img.icons8.com/clouds/500/healthy-food.png", width=150)
st.sidebar.title("🍽️ Nutrition Navigator")
app_mode = st.sidebar.radio("Navigate", ["🏠 Home", "🔍 Predict"])

# --- Pages ---
if app_mode == "🏠 Home":
    st.title("🧠 Calorie Estimator with AI")
    st.markdown("""
    Welcome to **Nutrition Navigator** — your smart assistant for food recognition and calorie estimation.  
    📷 Just upload an image of a fruit or vegetable, and our AI will:
    - Predict what it is  
    - Show you how many calories it contains per 100g  
    - Compare it with other common foods  
    
    Empower your nutrition knowledge — visually, instantly, and intelligently.
    """)
    st.image("home_img.jpg", use_container_width=True, caption="Eat smart with image-based calorie prediction")

elif app_mode == "🔍 Predict":
    st.title("📷 Image Prediction")
    uploaded_image = st.file_uploader("Upload your food image:", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        st.image(uploaded_image, caption="Your Uploaded Image", use_container_width=True)

        if st.button("Predict Now 🚀"):
            st.toast("Predicting...", icon="🤖")
            result_index = model_prediction(uploaded_image)
            with open("labels.txt") as f:
                labels = [line.strip() for line in f.readlines()]
            predicted_label = labels[result_index].lower()
            st.session_state.predicted_label = predicted_label

            calories = calorie_data.get(predicted_label)
            st.success(f"🟢 Prediction: **{predicted_label.capitalize()}**")
            if calories:
                st.info(f"🔥 Estimated Calories: **{calories} kcal per 100g**")
            else:
                st.warning("⚠️ Calorie data unavailable for this item.")

            # --- Calorie Comparison ---
            st.subheader("📊 Calorie Comparison")
            compare_items = [predicted_label] + [item for item in calorie_data if item != predicted_label][:2]
            comp_df = pd.DataFrame({
                "Item": compare_items,
                "Calories": [calorie_data[item] for item in compare_items]
            })
            fig = px.bar(comp_df, x="Item", y="Calories", color="Item", title="Calories per 100g")
            st.plotly_chart(fig, use_container_width=True)

            # --- Nutritional Breakdown ---
            st.subheader("🧪 Nutritional Breakdown")
            selected_df = nutrition_data[nutrition_data["Item"].isin(compare_items)]
            numeric_cols = selected_df.select_dtypes(include=['float64', 'int64']).columns
            st.dataframe(selected_df.style.format({col: "{:.1f}" for col in numeric_cols}))

    else:
        # --- Manual Comparison ---
        st.subheader("📊 Manual Calorie Comparison")
        compare_items = st.multiselect("Select items to compare:", list(calorie_data.keys()))
        
        if compare_items:
            comp_df = pd.DataFrame({
                "Item": compare_items,
                "Calories": [calorie_data[item] for item in compare_items]
            })
            fig = px.bar(comp_df, x="Item", y="Calories", color="Item", title="Calories per 100g")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🧪 Nutritional Breakdown")
            selected_df = nutrition_data[nutrition_data["Item"].isin(compare_items)]
            numeric_cols = selected_df.select_dtypes(include=['float64', 'int64']).columns
            st.dataframe(selected_df.style.format({col: "{:.1f}" for col in numeric_cols}))

# --- Custom Footer ---
st.markdown("""
    <hr style="border-top: 1px solid #555;">
    <p style='text-align: center; color: #aaa;'>Powered by <strong>Nutrition Navigator</strong> | Smart Food Insights © 2025</p>
""", unsafe_allow_html=True)




#cd Fruit_veg_webapp
# streamlit run main.py