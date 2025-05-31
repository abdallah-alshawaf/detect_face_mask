# Face Mask Detection Web Application

A modern, AI-powered web application for real-time face mask detection using YOLOv8 and React with intelligent mask compliance analysis.

## 🌟 Features

### **Core Functionality**
- **Modern React Frontend**: Beautiful, responsive UI with drag-and-drop image upload
- **Flask Backend API**: RESTful API for image processing and AI inference
- **Real-time Detection**: Fast face mask detection with confidence scoring
- **Intelligent Analysis**: Automatic mask compliance assessment with safety ratings

### **Detection Capabilities**
- **Three Detection Classes**:
  - ✅ **With Mask** (Green) - Properly worn face masks
  - ❌ **Without Mask** (Red) - No mask detected
  - ⚠️ **Incorrect Mask** (Orange) - Improperly worn masks
- **Individual Detection Details**: Bounding boxes, confidence scores, and coordinates
- **Batch Statistics**: Total detections, class breakdown, and compliance rates

### **Smart Analysis Features**
- **🛡️ Safety Level Assessment**: High/Medium/Low safety ratings based on compliance
- **📊 Compliance Rate Calculation**: Percentage of faces properly wearing masks
- **🎯 Detection Summary**: Clear status messages about mask compliance
- **📈 Detailed Statistics**: Individual detection breakdown with confidence scores

### **User Experience**
- **Interactive Results**: Side-by-side comparison with tabbed interface
- **Download Results**: Save processed images with detections
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Demo Mode**: Fallback functionality when AI model is unavailable
- **Real-time Feedback**: Instant processing status and results

## 🏗️ Architecture

```
Face Mask Detection Web App
├── frontend/                    # React application
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── Header.js       # App header with branding
│   │   │   ├── ImageUploader.js # Drag-drop upload interface
│   │   │   ├── ResultsDisplay.js # Results with analysis
│   │   │   ├── LoadingSpinner.js # Animated loading states
│   │   │   └── Footer.js       # App footer with info
│   │   ├── App.js              # Main application logic
│   │   └── index.js            # Entry point
│   └── package.json            # Dependencies
├── backend/                     # Flask API server
│   ├── app.py                  # Main Flask application
│   ├── uploads/                # Temporary upload storage
│   ├── outputs/                # Processed image storage
│   ├── requirements.txt        # Full Python dependencies
│   └── requirements_simple.txt # Essential dependencies only
└── src/                        # Original AI detection system
    ├── detect.py               # Enhanced detection logic
    └── utils.py                # Utility functions
```

## 📚 Complete Setup Tutorial

### **Step 1: System Requirements**

Before starting, ensure your system meets these requirements:

#### **Minimum Requirements:**
- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for downloading dependencies

#### **Recommended for AI Model:**
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for faster processing)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or better)

### **Step 2: Install Prerequisites**

#### **2.1 Install Python**

**Windows:**
1. Go to [python.org](https://python.org/downloads/)
2. Download Python 3.8+ (3.11 recommended, 3.13 supported)
3. **Important**: Check "Add Python to PATH" during installation
4. Verify installation:
   ```bash
   python --version
   pip --version
   ```

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### **2.2 Install Node.js**

**All Platforms:**
1. Go to [nodejs.org](https://nodejs.org/)
2. Download LTS version (v18+ recommended)
3. Install with default settings
4. Verify installation:
   ```bash
   node --version
   npm --version
   ```

#### **2.3 Install Git**

**Windows:**
- Download from [git-scm.com](https://git-scm.com/)
- Install with default settings

**macOS:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt install git
```

### **Step 3: Download the Project**

#### **3.1 Clone or Download**

**Option A: Using Git (Recommended)**
```bash
# Clone the repository
git clone <your-repository-url>
cd face-mask-detection

# Or if you have the project folder already
cd "Face Mask"  # Navigate to your project directory
```

**Option B: Download ZIP**
1. Download the project as ZIP file
2. Extract to a folder (e.g., `C:\Face-Mask-Detection` or `~/face-mask-detection`)
3. Open terminal/command prompt in that folder

#### **3.2 Verify Project Structure**
```bash
# You should see these folders:
ls -la  # Linux/Mac
dir     # Windows

# Expected output:
# frontend/
# backend/
# src/
# models/
# config/
# start_backend.bat
# start_frontend.bat
# README.md
```

### **Step 4: Backend Setup (Python/Flask)**

#### **4.1 Navigate to Backend Directory**
```bash
cd backend
```

#### **4.2 Create Virtual Environment (Recommended)**

**Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# You should see (venv) in your prompt
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# You should see (venv) in your prompt
```

#### **4.3 Install Python Dependencies**

**Option A: Essential Dependencies (Recommended for beginners)**
```bash
pip install -r requirements_simple.txt
```

**Option B: Full Dependencies**
```bash
pip install -r requirements.txt
```

**Option C: Manual Installation (if above fails)**
```bash
# Install core packages one by one
pip install Flask==2.3.3
pip install Flask-CORS==4.0.0
pip install opencv-python==4.8.1.78
pip install Pillow==10.0.1
pip install numpy==1.24.3

# Optional: For full AI functionality
pip install ultralytics==8.0.196
pip install torch torchvision
```

#### **4.4 Test Backend Installation**
```bash
# Test if packages are installed correctly
python -c "import flask, cv2, PIL; print('Backend dependencies OK!')"
```

#### **4.5 Start Backend Server**
```bash
# Make sure you're in the backend/ directory
python app.py

# You should see:
# * Running on http://0.0.0.0:5000
# * Debug mode: on
```

**Keep this terminal open!** The backend server needs to stay running.

### **Step 5: Frontend Setup (React/Node.js)**

#### **5.1 Open New Terminal**
Open a **new** terminal/command prompt window (keep the backend running in the first one).

#### **5.2 Navigate to Frontend Directory**
```bash
# From project root directory
cd frontend
```

#### **5.3 Install Node.js Dependencies**
```bash
# Install all React dependencies
npm install

# This may take 2-5 minutes depending on your internet speed
```

#### **5.4 Start Frontend Development Server**
```bash
# Start React development server
npm start

# You should see:
# Local:            http://localhost:3000
# On Your Network:  http://192.168.x.x:3000
```

Your default browser should automatically open to `http://localhost:3000`.

### **Step 6: Verify Everything Works**

#### **6.1 Check Both Servers**
You should now have:
- **Backend**: Running on `http://localhost:5000`
- **Frontend**: Running on `http://localhost:3000`

#### **6.2 Test the Application**
1. **Open your browser** to `http://localhost:3000`
2. **Check the header** - you should see "Face Mask Detection" with a status indicator
3. **Test file upload**:
   - Drag and drop an image with faces
   - Or click "Choose File" to select an image
   - Supported formats: JPG, PNG, GIF, BMP, WebP, TIFF

#### **6.3 Verify Detection**
- **With AI Model**: You'll see actual face mask detection
- **Demo Mode**: You'll see a demo detection with sample results
- **Results should show**:
  - Original vs processed image comparison
  - Mask detection summary with safety badges
  - Detailed statistics and compliance rates

### **Step 7: Using Automated Scripts (Alternative)**

If manual setup is complex, use the provided scripts:

#### **Windows:**
```bash
# Terminal 1: Start backend
start_backend.bat

# Terminal 2: Start frontend  
start_frontend.bat
```

#### **macOS/Linux:**
```bash
# Make scripts executable
chmod +x start_backend.sh start_frontend.sh

# Terminal 1: Start backend
./start_backend.sh

# Terminal 2: Start frontend
./start_frontend.sh
```

### **Step 8: Troubleshooting Common Issues**

#### **8.1 Python Issues**

**"Python not found":**
```bash
# Windows: Try these alternatives
py --version
python3 --version

# Add Python to PATH manually:
# Control Panel → System → Advanced → Environment Variables
# Add Python installation directory to PATH
```

**"pip not found":**
```bash
# Windows
python -m pip --version

# macOS/Linux
python3 -m pip --version
```

**Package installation fails:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Try installing with --user flag
pip install --user -r requirements_simple.txt

# For Python 3.13 compatibility issues:
pip install Flask Flask-CORS opencv-python-headless Pillow numpy
```

#### **8.2 Node.js Issues**

**"npm not found":**
- Reinstall Node.js from [nodejs.org](https://nodejs.org/)
- Restart your terminal after installation

**"npm install fails":**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and try again
rm -rf node_modules package-lock.json  # Linux/Mac
rmdir /s node_modules & del package-lock.json  # Windows

npm install
```

**Port 3000 already in use:**
```bash
# Kill process using port 3000
# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID_NUMBER> /F

# macOS/Linux:
lsof -ti:3000 | xargs kill -9

# Or use different port:
npm start -- --port 3001
```

#### **8.3 Backend Issues**

**"Module not found" errors:**
```bash
# Make sure virtual environment is activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Reinstall dependencies
pip install -r requirements_simple.txt
```

**"Port 5000 already in use":**
```bash
# Kill process using port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F

# macOS/Linux:
lsof -ti:5000 | xargs kill -9
```

**AI model not loading:**
- The app will automatically switch to **Demo Mode**
- You'll see "Demo Mode" indicators in the UI
- All functionality works, but with sample detection results

#### **8.4 Browser Issues**

**CORS errors:**
- Ensure both frontend (3000) and backend (5000) are running
- Try refreshing the page
- Check browser console for detailed error messages

**Images not displaying:**
- Check browser console for errors
- Ensure uploaded images are valid formats
- Try smaller image files (< 10MB)

### **Step 9: Advanced Configuration**

#### **9.1 GPU Acceleration (Optional)**

For faster AI processing with NVIDIA GPU:

```bash
# Check if you have CUDA-capable GPU
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install CUDA-enabled OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

#### **9.2 Custom Model Setup**

If you have a trained YOLOv8 model:

1. **Place your model** in `models/best.pt`
2. **Update configuration** in `backend/app.py`:
   ```python
   MODEL_PATH = '../models/your_model.pt'
   CONFIG_PATH = '../config/your_dataset.yaml'
   ```
3. **Restart the backend server**

#### **9.3 Production Deployment**

For production use:

```bash
# Build React app for production
cd frontend
npm run build

# Install production server
npm install -g serve
serve -s build -l 3000

# Use production WSGI server for backend
cd ../backend
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Step 10: Next Steps**

#### **10.1 Explore Features**
- **Upload different images** to test detection accuracy
- **Adjust confidence threshold** in advanced settings
- **Download processed images** with annotations
- **View detailed statistics** in the results tab

#### **10.2 Customize the Application**
- **Modify UI components** in `frontend/src/components/`
- **Add new API endpoints** in `backend/app.py`
- **Customize styling** in component CSS files
- **Add new detection classes** by training custom models

#### **10.3 Learn More**
- **React Documentation**: [reactjs.org](https://reactjs.org/)
- **Flask Documentation**: [flask.palletsprojects.com](https://flask.palletsprojects.com/)
- **YOLOv8 Documentation**: [docs.ultralytics.com](https://docs.ultralytics.com/)
- **Computer Vision Tutorials**: [opencv.org](https://opencv.org/)

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** (v16 or higher) - [Download](https://nodejs.org/)
- **Python** (3.8+ recommended, 3.13 supported) - [Download](https://python.org/)
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

# Install essential dependencies (recommended for Python 3.13)
pip install -r requirements_simple.txt

# Or install full dependencies
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

### **Getting Started**
1. **Access the Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

2. **Upload an Image**:
   - Drag and drop an image or click to select
   - Supported formats: JPG, PNG, GIF, BMP, WebP, TIFF
   - Adjust confidence threshold in advanced settings (optional)

### **Understanding Results**

3. **Mask Detection Summary**:
   - **🛡️ High Safety**: All faces properly wearing masks (Green badge)
   - **⚠️ Medium Safety**: Partial compliance (Orange badge)
   - **🚨 Low Safety**: Poor mask compliance (Red badge)

4. **Detailed Analysis**:
   - **Compliance Rate**: Percentage of faces with proper masks
   - **Individual Counts**: Breakdown by mask type
   - **Detection Details**: Bounding boxes and confidence scores

5. **Download & Share**:
   - Download processed images with annotations
   - View side-by-side comparison
   - Access detailed statistics

### **Detection Classes**
- **Green boxes**: Properly worn masks (compliant)
- **Red boxes**: No mask detected (non-compliant)
- **Orange boxes**: Incorrectly worn masks (non-compliant)

## 🔧 Configuration

### Backend Configuration

Edit `backend/app.py` to modify:

```python
# Model paths
MODEL_PATH = '../models/best.pt'  # Your trained model
CONFIG_PATH = '../config/dataset.yaml'  # Dataset config

# Image processing settings
max_size = 800  # Maximum image dimension for processing

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
Returns API status, model readiness, and demo mode status.

**Response:**
```json
{
  "status": "healthy",
  "detector_ready": true,
  "model_path": "../models/best.pt",
  "demo_mode": false
}
```

### Image Detection
```http
POST /api/detect
Content-Type: multipart/form-data

Parameters:
- image: Image file
- confidence: Confidence threshold (0.1-0.9)
```

**Response:**
```json
{
  "success": true,
  "file_id": "uuid",
  "original_image": "data:image/png;base64,...",
  "processed_image": "data:image/png;base64,...",
  "download_url": "/api/download/uuid",
  "statistics": {
    "total_detections": 3,
    "class_counts": {
      "with_mask": 2,
      "without_mask": 1,
      "mask_weared_incorrect": 0
    },
    "detections": [
      {
        "class": "with_mask",
        "confidence": 0.95,
        "bbox": [100, 50, 200, 150]
      }
    ]
  },
  "analysis": {
    "summary": "⚠️ 2/3 face(s) properly wearing masks",
    "compliance_rate": 66.7,
    "safety_level": "medium"
  }
}
```

### Download Result
```http
GET /api/download/{file_id}
```
Downloads the processed image with annotations.

### Cleanup
```http
DELETE /api/cleanup/{file_id}
```
Removes processed files from server.

## 🎨 UI Components

### **ImageUploader**
- **Drag-and-drop interface** with visual feedback
- **File preview** with metadata (size, type, dimensions)
- **Confidence threshold slider** (10%-90%)
- **Advanced settings panel** with explanations
- **Supported format indicators**

### **ResultsDisplay**
- **Tabbed interface**: Image Comparison & Detection Statistics
- **Side-by-side image comparison** with proper aspect ratios
- **Mask Detection Summary** with color-coded safety badges
- **Compliance details** with individual face counts
- **Download functionality** with one-click save
- **Detailed statistics table** with individual detections

### **LoadingSpinner**
- **Animated AI brain icon** with pulsing effect
- **Multi-ring spinner animation** with different speeds
- **Processing status messages**
- **Responsive sizing** (small, medium, large)

### **Header & Footer**
- **Modern gradient design** with animated backgrounds
- **Responsive navigation** with status indicators
- **Technology showcase** and feature highlights
- **Safety class indicators** with color coding

## 🔒 Security Features

- **File type validation** with whitelist approach
- **Secure filename handling** with UUID generation
- **Automatic file cleanup** after processing
- **CORS protection** for cross-origin requests
- **Input sanitization** and validation
- **Error handling** with graceful fallbacks

## 📱 Responsive Design

The application is fully responsive and works on:
- **Desktop** (1920px+): Full-featured interface with side-by-side layout
- **Tablet** (768px-1919px): Optimized layout with touch support
- **Mobile** (320px-767px): Simplified interface with stacked layout

### **Mobile Optimizations**
- Touch-friendly drag-and-drop areas
- Simplified navigation with collapsible sections
- Optimized image display for small screens
- Finger-friendly button sizes and spacing

## 🐛 Troubleshooting

### Common Issues

1. **Backend not starting**:
   ```bash
   # Check Python version (3.13 supported)
   python --version
   
   # Try simplified dependencies first
   pip install -r requirements_simple.txt
   
   # Or install individually
   pip install Flask Flask-CORS opencv-python Pillow numpy
   ```

2. **"No faces detected" but masks are visible**:
   - Check console logs for detection statistics
   - Verify model is loading correctly
   - Try adjusting confidence threshold
   - Ensure image quality is sufficient

3. **Frontend not starting**:
   ```bash
   # Check Node.js version
   node --version
   
   # Clear npm cache and reinstall
   npm cache clean --force
   rm -rf node_modules package-lock.json
   npm install
   ```

4. **Model not found**:
   - Ensure your trained model is in `models/best.pt`
   - Or update the `MODEL_PATH` in `backend/app.py`
   - System will fallback to demo mode if model unavailable

5. **Image dimension issues**:
   - Images are automatically resized to max 800px
   - Aspect ratios are preserved
   - Use `object-fit: contain` for proper display

6. **CORS errors**:
   - Ensure Flask-CORS is installed
   - Check that frontend proxy is configured correctly
   - Verify both servers are running on correct ports

### Performance Tips

1. **GPU Acceleration**:
   - Install PyTorch with CUDA support for faster inference
   - Ensure your GPU drivers are up to date
   - Check GPU utilization during processing

2. **Memory Usage**:
   - Large images are automatically resized to 800px max
   - Temporary files are cleaned up after processing
   - Monitor memory usage with multiple concurrent requests

3. **Network Issues**:
   - Check firewall settings for ports 3000 and 5000
   - Ensure both servers are running and accessible
   - Test API endpoints directly with curl

## 🔄 Development

### Adding New Features

1. **Backend**: Add new routes in `backend/app.py`
2. **Frontend**: Create new components in `frontend/src/components/`
3. **Styling**: Use CSS modules with responsive design
4. **Testing**: Test on multiple devices and browsers

### Building for Production

```bash
# Build React app
cd frontend
npm run build

# Serve with a production server
npm install -g serve
serve -s build -l 3000

# For backend, use a WSGI server like Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### **Demo Mode**
The application includes a demo mode that activates when:
- AI model is not available
- Dependencies are missing
- Model loading fails

Demo mode provides:
- Sample detection results
- Full UI functionality
- Educational value without requiring trained models

## 📊 Model Information

- **Architecture**: YOLOv8n (Nano) - Optimized for speed and accuracy
- **Input Size**: 640x640 pixels (auto-resized)
- **Classes**: 3 detection classes
  - `with_mask`: Properly worn face masks
  - `without_mask`: No mask detected
  - `mask_weared_incorrect`: Improperly worn masks
- **Performance**: ~85-90% mAP@0.5 on test dataset
- **Speed**: 30-50 FPS (depending on hardware)
- **Model Size**: ~6MB (YOLOv8n)

### **Detection Accuracy**
- **High Confidence** (>80%): Very reliable detections
- **Medium Confidence** (50-80%): Generally accurate
- **Low Confidence** (<50%): May require manual verification

## 🎯 Advanced Features

### **Mask Compliance Analysis**
- **Safety Level Assessment**: Automatic categorization of overall safety
- **Compliance Rate Calculation**: Mathematical analysis of mask adherence
- **Individual Face Tracking**: Detailed breakdown per detected face
- **Visual Indicators**: Color-coded badges and progress bars

### **Image Processing**
- **Automatic Format Conversion**: Supports all major image formats
- **Intelligent Resizing**: Maintains aspect ratios while optimizing for processing
- **Quality Preservation**: Minimal compression for accurate detection
- **Batch Processing Ready**: Architecture supports multiple image processing

### **API Enhancements**
- **Detailed Response Format**: Comprehensive detection information
- **Error Handling**: Graceful fallbacks and informative error messages
- **File Management**: Automatic cleanup and secure file handling
- **Logging**: Debug information for troubleshooting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper testing
4. Ensure responsive design works on all devices
5. Add documentation for new features
6. Submit a pull request with detailed description

### **Development Guidelines**
- Follow React best practices and hooks patterns
- Use semantic HTML and accessible design
- Implement proper error boundaries
- Test on multiple browsers and devices
- Document API changes and new endpoints

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:

1. **Check the troubleshooting section** above for common solutions
2. **Review console logs** for detailed error messages
3. **Verify dependencies** are properly installed and compatible
4. **Test API endpoints** directly to isolate frontend/backend issues
5. **Check model availability** and demo mode status

### **Getting Help**
- Console logs provide detailed debugging information
- API health endpoint shows system status
- Demo mode allows testing without full setup
- Responsive design adapts to your device automatically

## 🎯 Future Enhancements

### **Planned Features**
- [ ] **Real-time webcam detection** with live video processing
- [ ] **Batch image processing** for multiple files
- [ ] **User authentication** and session management
- [ ] **Detection history** with saved results
- [ ] **Model performance metrics** and analytics dashboard
- [ ] **Custom model upload** for specialized use cases
- [ ] **API rate limiting** and usage analytics
- [ ] **Docker containerization** for easy deployment

### **Advanced Capabilities**
- [ ] **Multi-language support** for international use
- [ ] **Accessibility improvements** for users with disabilities
- [ ] **Progressive Web App** features for mobile installation
- [ ] **Offline mode** with cached models
- [ ] **Advanced analytics** with compliance trends
- [ ] **Integration APIs** for third-party systems
