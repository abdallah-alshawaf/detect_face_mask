"""
Flask backend API for face mask detection
Handles image uploads and returns processed results
"""

import os
import sys
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import base64
import io

# Add src directory to path to import detection modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

try:
    from detect import FaceMaskDetector
    DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import FaceMaskDetector: {e}")
    print("The API will run in demo mode without actual detection.")
    DETECTOR_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize detector
MODEL_PATH = '../models/best.pt'  # Path to trained model
CONFIG_PATH = '../config/dataset.yaml'

# Fallback to demo model if trained model doesn't exist
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = '../yolov8n.pt'
    print("Using YOLOv8n pretrained model (demo mode)")

detector = None
if DETECTOR_AVAILABLE:
    try:
        detector = FaceMaskDetector(MODEL_PATH, CONFIG_PATH)
        print("Face mask detector initialized successfully")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        detector = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    # Ensure the image is in the correct format
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Convert BGR to RGB for proper display
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    else:
        # If already RGB or grayscale, use as is
        image_rgb = image_array

    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)

    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG', optimize=True)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def load_and_resize_image(image_path, max_size=800):
    """Load image and resize if too large while maintaining aspect ratio"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Get original dimensions
    height, width = image.shape[:2]

    # Resize if image is too large
    if max(height, width) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        image = cv2.resize(image, (new_width, new_height),
                           interpolation=cv2.INTER_AREA)

    return image


def analyze_detection_results(stats):
    """Analyze detection results to provide a summary"""
    total_detections = stats.get('total_detections', 0)
    class_counts = stats.get('class_counts', {})

    with_mask = class_counts.get('with_mask', 0)
    without_mask = class_counts.get('without_mask', 0)
    incorrect_mask = class_counts.get('mask_weared_incorrect', 0)

    if total_detections == 0:
        return {
            'summary': 'No faces detected in the image',
            'mask_compliance': 'unknown',
            'safety_level': 'unknown'
        }

    # Calculate compliance
    compliant_faces = with_mask
    non_compliant_faces = without_mask + incorrect_mask
    compliance_rate = (compliant_faces / total_detections) * \
        100 if total_detections > 0 else 0

    # Determine overall status
    if compliance_rate == 100:
        summary = f"✅ All {total_detections} face(s) properly wearing masks"
        safety_level = "high"
        mask_compliance = "full"
    elif compliance_rate >= 50:
        summary = f"⚠️ {compliant_faces}/{total_detections} face(s) properly wearing masks"
        safety_level = "medium"
        mask_compliance = "partial"
    else:
        summary = f"❌ Only {compliant_faces}/{total_detections} face(s) properly wearing masks"
        safety_level = "low"
        mask_compliance = "poor"

    return {
        'summary': summary,
        'mask_compliance': mask_compliance,
        'safety_level': safety_level,
        'compliance_rate': round(compliance_rate, 1),
        'details': {
            'total_faces': total_detections,
            'with_mask': with_mask,
            'without_mask': without_mask,
            'incorrect_mask': incorrect_mask
        }
    }


def create_demo_detection(image_path):
    """Create a demo detection result when the actual detector is not available"""
    # Load the image with proper resizing
    image = load_and_resize_image(image_path)

    # Create a simple demo annotation
    height, width = image.shape[:2]

    # Draw a demo bounding box (center of image)
    x1, y1 = width // 4, height // 4
    x2, y2 = 3 * width // 4, 3 * height // 4

    # Draw rectangle and text
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, 'DEMO: with_mask (0.95)', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Create demo statistics
    stats = {
        'total_detections': 1,
        'class_counts': {
            'with_mask': 1,
            'without_mask': 0,
            'mask_weared_incorrect': 0
        },
        'detections': [
            {
                'class': 'with_mask',
                'confidence': 0.95,
                'bbox': [x1, y1, x2, y2]
            }
        ]
    }

    return image, stats


def get_detailed_detection_stats(detector, image_path, conf_threshold):
    """Get detailed detection statistics including individual detections"""
    # Load and convert image
    image, converted_path = detector.load_and_convert_image(image_path)

    # Run inference
    results = detector.model(image, conf=conf_threshold, verbose=False)

    # Initialize stats
    class_counts = {class_name: 0 for class_name in detector.class_names}
    detections = []
    total_detections = 0

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id < len(detector.class_names):
                class_name = detector.class_names[class_id]
                class_counts[class_name] += 1
                total_detections += 1

                # Add individual detection details
                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

    return {
        'total_detections': total_detections,
        'class_counts': class_counts,
        'detections': detections
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_ready': detector is not None,
        'model_path': MODEL_PATH,
        'demo_mode': not DETECTOR_AVAILABLE
    })


@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """Main detection endpoint"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        input_filename = f"{file_id}_input.{file_extension}"
        output_filename = f"{file_id}_output.png"

        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Save uploaded file
        file.save(input_path)

        # Get confidence threshold from request
        conf_threshold = float(request.form.get('confidence', 0.5))

        # Process image
        if detector is not None:
            # Use actual detector
            print(f"Processing image with actual detector: {input_filename}")
            annotated_image = detector.detect_image(
                input_path, 
                conf_threshold=conf_threshold,
                save_path=output_path,
                auto_convert=True
            )
            
            # Get detailed detection statistics
            stats = get_detailed_detection_stats(detector, input_path, conf_threshold)
            print(f"Detection stats: {stats}")
        else:
            # Use demo detection
            print(f"Processing image with demo detector: {input_filename}")
            annotated_image, stats = create_demo_detection(input_path)
            cv2.imwrite(output_path, annotated_image)
            print(f"Demo stats: {stats}")
        
        # Convert images to base64 for response
        original_image = load_and_resize_image(input_path)
        original_b64 = image_to_base64(original_image)
        processed_b64 = image_to_base64(annotated_image)
        
        # Clean up input file
        os.remove(input_path)

        # Analyze detection results
        analysis = analyze_detection_results(stats)
        print(f"Analysis result: {analysis}")

        response = {
            'success': True,
            'file_id': file_id,
            'original_image': original_b64,
            'processed_image': processed_b64,
            'download_url': f'/api/download/{file_id}',
            'statistics': stats,
            'confidence_threshold': conf_threshold,
            'demo_mode': detector is None,
            'analysis': analysis
        }
        
        return jsonify(response)

    except Exception as e:
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)

        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/download/<file_id>', methods=['GET'])
def download_result(file_id):
    """Download processed image"""
    output_filename = f"{file_id}_output.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    if not os.path.exists(output_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"face_mask_detection_{file_id}.png",
        mimetype='image/png'
    )


@app.route('/api/cleanup/<file_id>', methods=['DELETE'])
def cleanup_files(file_id):
    """Clean up processed files"""
    output_filename = f"{file_id}_output.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    if os.path.exists(output_path):
        os.remove(output_path)
        return jsonify({'success': True, 'message': 'File cleaned up'})

    return jsonify({'success': False, 'message': 'File not found'})


if __name__ == '__main__':
    print("Starting Face Mask Detection API...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    if detector is None:
        print("⚠️  Running in DEMO MODE - actual detection not available")
        print("   Install dependencies and ensure src/detect.py is available for full functionality")
    app.run(debug=True, host='0.0.0.0', port=5000)
