# Face Mask Detection Web Application

A modern, AI-powered web application for real-time face mask detection using YOLOv8 and React.

## 🌟 Features

- **Modern React Frontend**: Beautiful, responsive UI with drag-and-drop image upload
- **Flask Backend API**: RESTful API for image processing and AI inference
- **Real-time Detection**: Fast face mask detection with confidence scoring
- **Three Detection Classes**:
  - ✅ With Mask (Green)
  - ❌ Without Mask (Red) 
  - ⚠️ Incorrect Mask (Orange)
- **Interactive Results**: Side-by-side comparison with detailed statistics
- **Download Results**: Save processed images with detections
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## 🏗️ Architecture

```
Face Mask Detection Web App
├── frontend/          # React application
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── App.js        # Main application
│   │   └── index.js      # Entry point
│   └── package.json      # Dependencies
├── backend/           # Flask API server
│   ├── app.py            # Main Flask application
│   ├── uploads/          # Temporary upload storage
│   ├── outputs/          # Processed image storage
│   └── requirements.txt  # Python dependencies
└── src/               # Original AI detection system
    ├── detect.py         # Detection logic
    └── utils.py          # Utility functions
```

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v16 or higher) - [Download](https://nodejs.org/)
- **Python** (3.8 or higher) - [Download](https://python.org/)
- **Git** - [Download](https://git-scm.com/)

### Option 1: Automated Setup (Recommended)

#### Windows:
```bash
# Start backend (in one terminal)
start_backend.bat

# Start frontend (in another terminal)
start_frontend.bat
```

#### Linux/Mac:
```bash
# Make scripts executable
chmod +x start_backend.sh start_frontend.sh

# Start backend (in one terminal)
./start_backend.sh

# Start frontend (in another terminal)
./start_frontend.sh
```

### Option 2: Manual Setup

#### Backend Setup:
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start Flask server
python app.py
```

#### Frontend Setup:
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start React development server
npm start
```

## 🌐 Usage

1. **Access the Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

2. **Upload an Image**:
   - Drag and drop an image or click to select
   - Supported formats: JPG, PNG, GIF, BMP, WebP, TIFF
   - Adjust confidence threshold (optional)

3. **View Results**:
   - Compare original vs processed images
   - View detailed detection statistics
   - Download the processed image

4. **Detection Classes**:
   - **Green boxes**: Properly worn masks
   - **Red boxes**: No mask detected
   - **Orange boxes**: Incorrectly worn masks

## 🔧 Configuration

### Backend Configuration

Edit `backend/app.py` to modify:

```python
# Model paths
MODEL_PATH = '../models/best.pt'  # Your trained model
CONFIG_PATH = '../config/dataset.yaml'  # Dataset config

# Server settings
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Frontend Configuration

Edit `frontend/package.json` to modify the proxy:

```json
{
  "proxy": "http://localhost:5000"
}
```

## 📡 API Endpoints

### Health Check
```http
GET /api/health
```
Returns API status and model readiness.

### Image Detection
```http
POST /api/detect
Content-Type: multipart/form-data

Parameters:
- image: Image file
- confidence: Confidence threshold (0.1-0.9)
```

### Download Result
```http
GET /api/download/{file_id}
```
Downloads the processed image.

### Cleanup
```http
DELETE /api/cleanup/{file_id}
```
Removes processed files from server.

## 🎨 UI Components

### ImageUploader
- Drag-and-drop interface
- File preview with metadata
- Confidence threshold slider
- Advanced settings panel

### ResultsDisplay
- Tabbed interface (Comparison/Statistics)
- Side-by-side image comparison
- Detailed detection breakdown
- Download functionality

### LoadingSpinner
- Animated AI brain icon
- Multi-ring spinner animation
- Processing status messages

## 🔒 Security Features

- File type validation
- Secure filename handling
- Temporary file cleanup
- CORS protection
- Input sanitization

## 📱 Responsive Design

The application is fully responsive and works on:
- **Desktop**: Full-featured interface
- **Tablet**: Optimized layout with touch support
- **Mobile**: Simplified interface for small screens

## 🐛 Troubleshooting

### Common Issues

1. **Backend not starting**:
   ```bash
   # Check Python version
   python --version
   
   # Install dependencies manually
   pip install flask flask-cors opencv-python ultralytics
   ```

2. **Frontend not starting**:
   ```bash
   # Check Node.js version
   node --version
   
   # Clear npm cache
   npm cache clean --force
   npm install
   ```

3. **Model not found**:
   - Ensure your trained model is in `models/best.pt`
   - Or update the `MODEL_PATH` in `backend/app.py`

4. **CORS errors**:
   - Ensure Flask-CORS is installed
   - Check that frontend proxy is configured correctly

### Performance Tips

1. **GPU Acceleration**:
   - Install PyTorch with CUDA support for faster inference
   - Ensure your GPU drivers are up to date

2. **Memory Usage**:
   - Large images are automatically resized
   - Temporary files are cleaned up after processing

3. **Network Issues**:
   - Check firewall settings for ports 3000 and 5000
   - Ensure both servers are running

## 🔄 Development

### Adding New Features

1. **Backend**: Add new routes in `backend/app.py`
2. **Frontend**: Create new components in `frontend/src/components/`
3. **Styling**: Use CSS modules or styled-components

### Building for Production

```bash
# Build React app
cd frontend
npm run build

# Serve with a production server
npm install -g serve
serve -s build -l 3000
```

## 📊 Model Information

- **Architecture**: YOLOv8n (Nano)
- **Input Size**: 640x640 pixels
- **Classes**: 3 (with_mask, without_mask, mask_weared_incorrect)
- **Performance**: ~85-90% mAP@0.5
- **Speed**: 30-50 FPS (depending on hardware)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the console logs for error messages
3. Ensure all dependencies are properly installed
4. Verify that both frontend and backend servers are running

## 🎯 Future Enhancements

- [ ] Real-time webcam detection
- [ ] Batch image processing
- [ ] User authentication
- [ ] Detection history
- [ ] Model performance metrics
- [ ] Custom model upload
- [ ] API rate limiting
- [ ] Docker containerization

---

**Made with ❤️ for public safety and AI education** 