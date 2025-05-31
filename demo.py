"""
Demo script for face mask detection
Downloads a pre-trained model and demonstrates detection capabilities
"""

import os
import cv2
import numpy as np
import requests
from pathlib import Path
import argparse
from ultralytics import YOLO
import urllib.request


class FaceMaskDemo:
    def __init__(self):
        """Initialize the face mask detection demo"""
        self.model = None
        self.class_names = ['with_mask',
                            'without_mask', 'mask_weared_incorrect']
        self.colors = {
            'with_mask': (0, 255, 0),          # Green
            'without_mask': (0, 0, 255),       # Red
            'mask_weared_incorrect': (0, 165, 255)  # Orange
        }

    def download_pretrained_model(self):
        """Download a pre-trained YOLOv8 model for demonstration"""
        print("Downloading pre-trained YOLOv8 model...")

        # Create models directory
        os.makedirs('models', exist_ok=True)

        # For demo purposes, we'll use a general YOLOv8 model
        # In a real scenario, you would use your trained face mask model
        model_path = 'models/yolov8n.pt'

        if not os.path.exists(model_path):
            print("Downloading YOLOv8n model...")
            self.model = YOLO('yolov8n.pt')  # This will auto-download
            print("Model downloaded successfully!")
        else:
            self.model = YOLO(model_path)
            print("Using existing model.")

        return model_path

    def create_sample_images(self):
        """Create sample images for demonstration"""
        print("Creating sample images for demonstration...")

        # Create sample images directory
        os.makedirs('sample_images', exist_ok=True)

        # Create simple colored rectangles as sample images
        sample_images = []

        for i, (name, color) in enumerate(self.colors.items()):
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            img[:] = color

            # Add text
            text = f"Sample {name.replace('_', ' ').title()}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness)

            # Center text
            x = (img.shape[1] - text_width) // 2
            y = (img.shape[0] + text_height) // 2

            cv2.putText(img, text, (x, y), font, font_scale,
                        (255, 255, 255), thickness)

            # Save image
            img_path = f'sample_images/sample_{name}.jpg'
            cv2.imwrite(img_path, img)
            sample_images.append(img_path)
            print(f"Created: {img_path}")

        return sample_images

    def detect_faces_general(self, image, conf_threshold=0.5):
        """
        Detect faces using general YOLO model (for demo purposes)
        In a real scenario, this would use the trained face mask model
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)

        # Filter for person class (class 0 in COCO dataset)
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id == 0:  # Person class
                    detections.append({
                        'box': box,
                        'confidence': conf,
                        'class': 'person'
                    })

        return detections

    def annotate_image_demo(self, image, detections):
        """Annotate image with demo detections"""
        annotated = image.copy()

        for detection in detections:
            box = detection['box']
            conf = detection['confidence']

            x1, y1, x2, y2 = box.astype(int)

            # For demo, randomly assign mask status
            import random
            mask_status = random.choice(list(self.colors.keys()))
            color = self.colors[mask_status]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"{mask_status.replace('_', ' ').title()}: {conf:.2f}"

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw label background
            cv2.rectangle(annotated, (x1, y1 - text_height - baseline - 5),
                          (x1 + text_width, y1), color, -1)

            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated

    def demo_webcam(self, camera_id=0):
        """Demonstrate face mask detection using webcam"""
        print("Starting webcam demo...")
        print("Note: This is a demonstration using a general object detection model.")
        print("For actual face mask detection, please train the model with the provided dataset.")
        print("Press 'q' to quit.")

        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break

                # Detect faces/persons
                detections = self.detect_faces_general(frame)

                # Annotate frame
                annotated_frame = self.annotate_image_demo(frame, detections)

                # Add demo info
                info_text = "DEMO MODE - Using general object detection"
                cv2.putText(annotated_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Display frame
                cv2.imshow('Face Mask Detection Demo', annotated_frame)

                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("Demo completed.")

    def demo_image(self, image_path):
        """Demonstrate face mask detection on an image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Processing image: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Detect faces/persons
        detections = self.detect_faces_general(image)

        # Annotate image
        annotated_image = self.annotate_image_demo(image, detections)

        # Add demo info
        info_text = "DEMO MODE"
        cv2.putText(annotated_image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display image
        cv2.imshow('Face Mask Detection Demo', annotated_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return annotated_image

    def run_demo(self, mode='webcam', image_path=None):
        """Run the demonstration"""
        print("=" * 50)
        print("Face Mask Detection Demo")
        print("=" * 50)
        print()
        print("This is a demonstration of the face mask detection system.")
        print("To use the full functionality, please:")
        print("1. Download the dataset from Kaggle")
        print("2. Run data preparation: python src/data_preparation.py")
        print("3. Train the model: python src/train.py")
        print("4. Use detection: python src/detect.py")
        print()

        # Download/load model
        self.download_pretrained_model()

        if mode == 'webcam':
            self.demo_webcam()
        elif mode == 'image' and image_path:
            self.demo_image(image_path)
        elif mode == 'samples':
            # Create and process sample images
            sample_images = self.create_sample_images()
            for img_path in sample_images:
                print(f"\nProcessing: {img_path}")
                self.demo_image(img_path)
        else:
            print("Invalid mode or missing image path")


def main():
    parser = argparse.ArgumentParser(description='Face mask detection demo')
    parser.add_argument('--mode', type=str, choices=['webcam', 'image', 'samples'],
                        default='samples', help='Demo mode')
    parser.add_argument('--image', type=str,
                        help='Path to image file (for image mode)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera ID for webcam mode')

    args = parser.parse_args()

    # Create demo instance
    demo = FaceMaskDemo()

    try:
        if args.mode == 'webcam':
            demo.run_demo('webcam')
        elif args.mode == 'image':
            if not args.image:
                print("Error: --image argument required for image mode")
                return
            demo.run_demo('image', args.image)
        else:  # samples
            demo.run_demo('samples')

    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
