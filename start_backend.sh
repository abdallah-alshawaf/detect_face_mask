#!/bin/bash

echo "Starting Face Mask Detection Backend..."
echo

cd backend

echo "Installing essential Python dependencies..."
pip install -r requirements_simple.txt

if [ $? -ne 0 ]; then
    echo
    echo "Warning: Some packages failed to install. Trying individual installation..."
    pip install Flask Flask-CORS opencv-python Pillow numpy
fi

echo
echo "Starting Flask backend server..."
echo "Backend API will be available at: http://localhost:5000"
echo "Note: If detection modules are missing, the server will run in DEMO mode"
echo

python app.py 