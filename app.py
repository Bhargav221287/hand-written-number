import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Handwritten Digit Recognition", layout="wide")

def main():
    st.title("Handwritten Digit Recognition")
    st.write("Capture a handwritten digit using your camera and get it identified!")
    
    # Backend API URL
    backend_url = st.text_input("Backend API URL", "http://localhost:5000/predict")
    
    # Create tabs for different input methods
    tab1, tab2, tab3 = st.tabs(["Camera Capture", "Upload Image", "Draw Digit"])
    
    with tab1:
        camera_input(backend_url)
    
    with tab2:
        upload_input(backend_url)
        
    with tab3:
        drawing_input(backend_url)

def camera_input(backend_url):
    st.subheader("Camera Input")
    
    # Camera capture using streamlit
    camera_image = st.camera_input("Take a picture of your handwritten digit")
    
    if camera_image is not None:
        process_image(camera_image, backend_url)

def upload_input(backend_url):
    st.subheader("Upload Image")
    
    # File uploader for images
    uploaded_image = st.file_uploader("Upload an image of a handwritten digit", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        process_image(uploaded_image, backend_url)

def drawing_input(backend_url):
    st.subheader("Draw Digit")
    
    # Use streamlit canvas for drawing
    try:
        from streamlit_drawable_canvas import st_canvas
        
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        if st.button("Recognize Drawn Digit"):
            if canvas_result.image_data is not None:
                # Convert the drawn image to grayscale and resize to 28x28
                image = canvas_result.image_data
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                image = cv2.resize(image, (28, 28))
                
                # Display the processed image
                st.image(image, caption="Processed Image (28x28)", width=150)
                
                # Convert to format expected by backend (flatten to 1D array)
                image_array = image.flatten().tolist()
                
                # Send to backend and display result
                send_to_backend(image_array, backend_url)
                
    except ImportError:
        st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
        st.code("pip install streamlit-drawable-canvas")

def process_image(image_data, backend_url):
    # Display the original image
    image = Image.open(image_data)
    st.image(image, caption="Original Image", width=300)
    
    # Process the image
    img_array = np.array(image)
    
    # Convert to grayscale if it's a color image
    if len(img_array.shape) > 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply thresholding to make digit more prominent
    _, img_array = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to locate the digit
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (assuming it's the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the digit and add padding
        padding = 5
        digit = img_array[max(0, y-padding):min(img_array.shape[0], y+h+padding), 
                         max(0, x-padding):min(img_array.shape[1], x+w+padding)]
        
        # Resize to 28x28 pixels
        digit = cv2.resize(digit, (28, 28))
        
        # Display the processed image
        st.image(digit, caption="Processed Image (28x28)", width=150)
        
        # Convert to format expected by backend (flatten to 1D array)
        digit_array = digit.flatten().tolist()
        
        # Send to backend and display result
        send_to_backend(digit_array, backend_url)
    else:
        st.error("No digit detected in the image. Please try again with a clearer image.")

def send_to_backend(image_array, backend_url):
    try:
        with st.spinner("Sending to backend for prediction..."):
            # Create payload for backend
            payload = {"image": image_array}
            
            # Send POST request to backend
            response = requests.post(backend_url, json=payload)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Display the result
                st.success(f"Prediction successful!")
                st.subheader(f"Predicted Digit: {result.get('prediction')}")
                
                # If confidence scores are included in the response
                if "confidence" in result:
                    st.write("Confidence Scores:")
                    for digit, score in enumerate(result["confidence"]):
                        st.progress(score)
                        st.write(f"Digit {digit}: {score:.4f}")
            else:
                st.error(f"Error from backend: {response.status_code}")
                st.code(response.text)
    except Exception as e:
        st.error(f"Failed to connect to backend: {str(e)}")
        st.info("Make sure your backend server is running and the URL is correct.")

if __name__ == "__main__":
    main()
