import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
import os

app = Flask(__name__)

# Global variable to store the model
model = None

def load_model():
    """
    Load the pre-trained model or train a new one if it doesn't exist
    """
    global model
    model_path = 'digit_recognition_model.h5'
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = tf.keras.models.load_model(model_path)
        return
    
    print("Training new model...")
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize and reshape data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(x_train, y_train, epochs=5, batch_size=128,
              validation_data=(x_test, y_test))
    
    # Save the model
    model.save(model_path)
    print(f"Model saved to {model_path}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict digit from image data
    """
    try:
        # Get data from request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Convert to numpy array and reshape
        image = np.array(data['image'], dtype=np.float32)
        
        # Check if the array has the right size
        if image.size != 28*28:
            return jsonify({'error': f'Expected 784 pixels, got {image.size}'}), 400
        
        # Normalize pixel values to [0,1]
        image = image / 255.0
        
        # Reshape to model input shape (1, 28, 28, 1)
        image = image.reshape(1, 28, 28, 1)
        
        # Make prediction
        predictions = model.predict(image)[0]
        predicted_digit = np.argmax(predictions)
        confidence = predictions.tolist()
        
        # Return prediction and confidence
        return jsonify({
            'prediction': int(predicted_digit),
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint to check if the service is running
    """
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Load the model at startup
    load_model()
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)
