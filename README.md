
<div align="center">
  <img src="https://github.com/user-attachments/assets/572c437b-eff3-4603-b9f1-8679d4ac859a" alt="face-mask-icon" width="120" height="120" style="margin-top:3px;">
  <h1>Face Mask Detection Web Application</h1>
  A modern, AI-powered web application for real-time face mask detection using YOLOv8 and React with intelligent mask compliance analysis.

</div>

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

## 🚀 Setup Tutorial

### **Prerequisites**
- **Node.js** (v16+) - [Download](https://nodejs.org/)
- **Python** (3.8+, 3.13 supported) - [Download](https://python.org/)
- **Git** - [Download](https://git-scm.com/)

### **Quick Setup (Automated)**

#### Windows:
```bash
# Terminal 1: Start backend
start_backend.bat

# Terminal 2: Start frontend
start_frontend.bat
```

#### Linux/Mac:
```bash
chmod +x start_backend.sh start_frontend.sh
./start_backend.sh    # Terminal 1
./start_frontend.sh   # Terminal 2
```

### **Manual Setup**

#### 1. Backend Setup:
```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements_simple.txt

# Start server
python app.py
```

#### 2. Frontend Setup:
```bash
cd frontend
npm install
npm start
```

#### 3. Access Application:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:5000

### **Common Issues**

**Python not found:**
```bash
# Windows alternatives
py --version
python3 --version
```

**Package installation fails:**
```bash
pip install Flask Flask-CORS opencv-python Pillow numpy
```

**Port already in use:**
```bash
# Kill process on port 3000/5000
# Windows: taskkill /PID <PID> /F
# Linux/Mac: lsof -ti:3000 | xargs kill -9
```

**Demo Mode**: App automatically switches to demo mode if AI model unavailable - all features work with sample results.

## 🤖 AI Model Training

### **Dataset Download**

**Kaggle Face Mask Dataset:**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API (get kaggle.json from kaggle.com/account)
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d andrewmvd/face-mask-detection
unzip face-mask-detection.zip -d dataset/
```

**Alternative Sources:**
- **Roboflow**: Pre-processed YOLO format datasets
- **Custom Collection**: Use `labelImg` tool for manual annotation
- **Public Datasets**: WIDER FACE, MAFA, CelebA with masks

### **Data Preparation**

```bash
# Convert PASCAL VOC XML to YOLO format
python src/convert_annotations.py --input dataset/annotations --output dataset/labels

# Split dataset (train/val/test)
python src/split_dataset.py --ratio 0.7 0.2 0.1

# Verify dataset structure:
# dataset/
# ├── images/
# │   ├── train/
# │   ├── val/
# │   └── test/
# └── labels/
#     ├── train/
#     ├── val/
#     └── test/
```

### **Model Training**

**Quick Training:**
```bash
# Install training dependencies
pip install ultralytics wandb

# Train YOLOv8 model
yolo train data=config/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640

# Monitor training
tensorboard --logdir runs/train
```

**Advanced Training:**
```bash
# Custom training script
python train.py \
  --data config/dataset.yaml \
  --weights yolov8n.pt \
  --epochs 100 \
  --batch-size 16 \
  --img-size 640 \
  --device 0  # GPU device

# Resume training
yolo train resume model=runs/train/exp/weights/last.pt
```

### **Dataset Configuration**

Create `config/dataset.yaml`:
```yaml
# Dataset paths
path: ../dataset
train: images/train
val: images/val
test: images/test

# Classes
nc: 3  # number of classes
names: ['with_mask', 'without_mask', 'mask_weared_incorrect']
```

### **Training Tips**

**Hardware Requirements:**
- **GPU**: NVIDIA GTX 1060+ (4GB+ VRAM)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB+ for dataset and models

**Optimization:**
```bash
# Faster training with mixed precision
yolo train data=config/dataset.yaml model=yolov8n.pt epochs=100 amp=True

# Data augmentation
yolo train data=config/dataset.yaml model=yolov8n.pt \
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=10 translate=0.1 scale=0.5 \
  flipud=0.5 fliplr=0.5 mosaic=1.0
```

**Model Evaluation:**
```bash
# Validate trained model
yolo val model=runs/train/exp/weights/best.pt data=config/dataset.yaml

# Test on custom images
yolo predict model=runs/train/exp/weights/best.pt source=test_images/
```

### **Expected Results**
- **Training Time**: 2-4 hours (GPU), 8-12 hours (CPU)
- **Dataset Size**: ~1700 images, 3 classes
- **Performance**: 85-90% mAP@0.5
- **Model Size**: ~6MB (YOLOv8n)

**Final Model**: Copy `runs/train/exp/weights/best.pt` to `models/best.pt` for web app integration.

## 🌐 Usage

1. **Upload Image**: Drag & drop or click to select (JPG, PNG, GIF, BMP, WebP, TIFF)
2. **View Results**: 
   - **🛡️ High Safety**: All faces with masks (Green)
   - **⚠️ Medium Safety**: Partial compliance (Orange)  
   - **🚨 Low Safety**: Poor compliance (Red)
3. **Download**: Save processed images with annotations

## 🔧 Configuration

### Backend (`backend/app.py`):
```python
MODEL_PATH = '../models/best.pt'  # Your trained model
max_size = 800  # Max image dimension
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Frontend (`frontend/package.json`):
```json
{ "proxy": "http://localhost:5000" }
```

## 📡 API Endpoints

### Health Check
```http
GET /api/health
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
  "original_image": "data:image/png;base64,...",
  "processed_image": "data:image/png;base64,...",
  "statistics": {
    "total_detections": 3,
    "class_counts": {"with_mask": 2, "without_mask": 1}
  },
  "analysis": {
    "summary": "⚠️ 2/3 face(s) properly wearing masks",
    "compliance_rate": 66.7,
    "safety_level": "medium"
  }
}
```

## 🎨 UI Components

- **ImageUploader**: Drag-drop interface with confidence slider
- **ResultsDisplay**: Tabbed comparison with safety badges
- **LoadingSpinner**: Animated AI brain with progress rings
- **Header/Footer**: Modern gradient design with status indicators

## 🔒 Security & Performance

- File type validation with UUID naming
- Automatic cleanup and CORS protection
- Responsive design (desktop/tablet/mobile)
- GPU acceleration support (optional)
- Images auto-resized to 800px max

## 📊 Model Information

- **Architecture**: YOLOv8n (6MB, 30-50 FPS)
- **Classes**: with_mask, without_mask, mask_weared_incorrect
- **Performance**: ~85-90% mAP@0.5
- **Input**: 640x640 pixels (auto-resized)

## 📷 Photos from the App and The Model
Labels from AI Training
![val_batch2_labels](https://github.com/user-attachments/assets/b20fd1f3-9033-4b08-a4fc-f84bc5c134b8)

Confidence Curve around 80%
![F1_curve](https://github.com/user-attachments/assets/7ebb6048-d352-4abf-9323-1612fcf0b3fb)

Running the model directly from python script
![Capture](https://github.com/user-attachments/assets/7a5d896d-0808-4f87-86f3-45f7feba99f0)

Front End React Website 
![screencapture-localhost-3000-2025-05-31-20_07_13](https://github.com/user-attachments/assets/60d4f0f0-065a-40fc-bdd5-926ceb0d2f8c)
