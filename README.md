# Plant Disease Detection API

This is a FastAPI application that uses a TensorFlow model to detect plant diseases from images.

## Features

- Image upload and processing
- Disease detection using a pre-trained TensorFlow model
- Detailed disease information including causes, symptoms, treatments, and prevention tips

## Deployment on Render

### Prerequisites

- A Render account
- Git repository with your code

### Steps to Deploy

1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. Log in to your Render account
3. Click on "New" and select "Web Service"
4. Connect your Git repository
5. Configure the following settings:
   - Name: plant-disease-detection (or your preferred name)
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app -k uvicorn.workers.UvicornWorker`
6. Click "Create Web Service"

### Important Notes

- Make sure your model file (`trained_plant_disease_model.keras`) is included in the repository
- The `data.json` file must be included in the repository
- The application will be available at the URL provided by Render after deployment

## API Usage

### Endpoint: `/predict`

**Method**: POST

**Request**: Form data with an image file

**Response**: JSON object with detailed disease information

Example:
```
curl -X POST "https://your-render-url.com/predict" -F "file=@plant_image.jpg"
```

## Local Development

To run the application locally:

```
uvicorn app:app --reload
```

The API will be available at http://127.0.0.1:8000
