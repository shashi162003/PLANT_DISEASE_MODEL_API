import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import json

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model (replace with your model's path)
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Load the disease data from data.json
with open('data.json', 'r') as f:
    disease_data = json.load(f)

# Define the list of class names (disease names) in the order matching the model's output
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Get the model's expected input shape (e.g., (224, 224, 3))
input_shape = model.input_shape[1:3]  # Extract height and width

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# Helper function to preprocess the image
def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize image to match model input shape
    image = image.resize(input_shape)
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # No normalization: assume model expects [0, 255]
    return img_array

# Helper function to format disease name
def format_disease_name(disease_name: str) -> str:
    # Replace underscores with spaces and apply title case
    formatted_name = disease_name.replace('___', ' ').replace('_', ' ').title()
    return formatted_name

# Helper function to get disease data from data.json
def get_disease_data(disease_name: str):
    formatted_name = format_disease_name(disease_name)
    
    # Create a mapping dictionary for disease names
    disease_mapping = {
        'Apple Apple Scab': 'Apple Scab',
        'Apple Black Rot': 'Apple Black Rot',
        'Apple Cedar Apple Rust': 'Apple Cedar Apple Rust',
        'Apple Healthy': 'Apple Healthy',
        'Blueberry Healthy': 'Blueberry Healthy',
        'Cherry Including Sour Powdery Mildew': 'Cherry Powdery Mildew',
        'Cherry Including Sour Healthy': 'Cherry Healthy',
        'Corn Maize Cercospora Leaf Spot Gray Leaf Spot': 'Corn Gray Leaf Spot',
        'Corn Maize Common Rust': 'Corn Common Rust',
        'Corn Maize Northern Leaf Blight': 'Corn Northern Leaf Blight',
        'Corn Maize Healthy': 'Corn Healthy',
        'Grape Black Rot': 'Grape Black Rot',
        'Grape Esca Black Measles': 'Grape Black Measles',
        'Grape Leaf Blight Isariopsis Leaf Spot': 'Grape Leaf Blight',
        'Grape Healthy': 'Grape Healthy',
        'Orange Haunglongbing Citrus Greening': 'Orange Haunglongbing (Citrus Greening)',
        'Peach Bacterial Spot': 'Peach Bacterial Spot',
        'Peach Healthy': 'Peach Healthy',
        'Pepper Bell Bacterial Spot': 'Pepper Bell Bacterial Spot',
        'Pepper Bell Healthy': 'Pepper Bell Healthy',
        'Potato Early Blight': 'Potato Early Blight',
        'Potato Late Blight': 'Potato Late Blight',
        'Potato Healthy': 'Potato Healthy',
        'Raspberry Healthy': 'Raspberry Healthy',
        'Soybean Healthy': 'Soybean Healthy',
        'Squash Powdery Mildew': 'Squash Powdery Mildew',
        'Strawberry Leaf Scorch': 'Strawberry Leaf Scorch',
        'Strawberry Healthy': 'Strawberry Healthy',
        'Tomato Bacterial Spot': 'Tomato Bacterial Spot',
        'Tomato Early Blight': 'Tomato Early Blight',
        'Tomato Late Blight': 'Tomato Late Blight',
        'Tomato Leaf Mold': 'Tomato Leaf Mold',
        'Tomato Septoria Leaf Spot': 'Tomato Septoria Leaf Spot',
        'Tomato Spider Mites Two Spotted Spider Mite': 'Tomato Spider Mites',
        'Tomato Target Spot': 'Tomato Target Spot',
        'Tomato Tomato Yellow Leaf Curl Virus': 'Tomato Yellow Leaf Curl Virus',
        'Tomato Tomato Mosaic Virus': 'Tomato Mosaic Virus',
        'Tomato Healthy': 'Tomato Healthy'
    }
    
    # Try to get the mapped name from our dictionary
    data_key = disease_mapping.get(formatted_name)
    
    # If we found a mapping, use it to get the data
    if data_key and data_key in disease_data:
        return disease_data[data_key]
    
    # If no direct mapping, try a more flexible approach
    for key in disease_data.keys():
        # Convert both to lowercase and remove special characters for comparison
        if key.lower().replace('(', '').replace(')', '').replace('-', ' ') in formatted_name.lower().replace('-', ' '):
            return disease_data[key]
        if formatted_name.lower().replace('-', ' ') in key.lower().replace('(', '').replace(')', '').replace('-', ' '):
            return disease_data[key]
    
    # If still not found, return a basic response with just the disease name
    return {"Disease Name": formatted_name}

# Define the prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check file extension
    if not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG, JPEG, or PNG.")

    try:
        # Read the uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Preprocess the image
        img_array = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])  # Get index of highest probability
        disease_name = CLASS_NAMES[predicted_index]  # Map index to disease name
        
        # Get the full disease data from data.json
        disease_info = get_disease_data(disease_name)

        # Return the complete disease information as JSON
        return JSONResponse(content=disease_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Add a root endpoint for health check
@app.get("/")
async def root():
    return {"status": "healthy", "message": "Plant Disease Detection API is running"}

# Add this at the end of the file
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the application
    uvicorn.run("app:app", host="0.0.0.0", port=port)