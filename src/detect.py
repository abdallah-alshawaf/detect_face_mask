"""
Detection script for face mask detection
Supports inference on images, videos, and webcam
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
import torch

from utils import load_config


class FaceMaskDetector:
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize face mask detector

        Args:
            model_path: Path to trained YOLO model
            config_path: Path to dataset configuration file (optional)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        self.model = YOLO(model_path)

        # Load class names
        if config_path and os.path.exists(config_path):
            config = load_config(config_path)
            self.class_names = config['names']
        else:
            # Default class names
            self.class_names = ['with_mask',
                                'without_mask', 'mask_weared_incorrect']

        # Define colors for each class
        self.colors = {
            'with_mask': (0, 255, 0),          # Green
            'without_mask': (0, 0, 255),       # Red
            'mask_weared_incorrect': (0, 165, 255)  # Orange
        }

        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.class_names}")

    def detect_image(self, image_path: str, conf_threshold: float = 0.5, save_path: str = None):
        """
        Detect face masks in a single image

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            save_path: Path to save output image (optional)

        Returns:
            Annotated image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)

        # Annotate image
        annotated_image = self.annotate_image(image.copy(), results[0])

        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Output saved to: {save_path}")

        return annotated_image

    def detect_video(self, video_path: str, conf_threshold: float = 0.5, save_path: str = None):
        """
        Detect face masks in a video

        Args:
            video_path: Path to input video
            conf_threshold: Confidence threshold for detections
            save_path: Path to save output video (optional)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Setup video writer if save path provided
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Run inference
                results = self.model(frame, conf=conf_threshold, verbose=False)

                # Annotate frame
                annotated_frame = self.annotate_image(frame, results[0])

                # Add frame info
                info_text = f"Frame: {frame_count + 1}/{total_frames}"
                cv2.putText(annotated_frame, info_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Save frame if writer available
                if writer:
                    writer.write(annotated_frame)

                # Display frame
                cv2.imshow('Face Mask Detection', annotated_frame)

                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

                # Print progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    print(
                        f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} FPS)")

        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        print(
            f"Video processing completed: {frame_count} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")

        if save_path:
            print(f"Output saved to: {save_path}")

    def detect_webcam(self, camera_id: int = 0, conf_threshold: float = 0.5):
        """
        Detect face masks using webcam

        Args:
            camera_id: Camera device ID
            conf_threshold: Confidence threshold for detections
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        print("Starting webcam detection. Press 'q' to quit.")
        print(
            "Note: If GUI display fails, frames will be saved to 'webcam_frames/' directory")

        frame_count = 0
        start_time = time.time()
        gui_available = True
        save_dir = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break

                # Run inference
                results = self.model(frame, conf=conf_threshold, verbose=False)

                # Annotate frame
                annotated_frame = self.annotate_image(frame, results[0])

                # Add FPS info
                if frame_count > 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(annotated_frame, fps_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Try to display frame
                if gui_available:
                    try:
                        cv2.imshow('Face Mask Detection - Webcam',
                                   annotated_frame)

                        # Check for exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error as e:
                        print(f"GUI display failed: {e}")
                        print("Switching to save mode - frames will be saved instead")
                        gui_available = False

                        # Create save directory
                        save_dir = Path("webcam_frames")
                        save_dir.mkdir(exist_ok=True)
                        print(f"Frames will be saved to: {save_dir}")

                # If GUI not available, save frames periodically
                if not gui_available:
                    # Save every 30 frames (~1 second at 30fps)
                    if frame_count % 30 == 0:
                        frame_filename = save_dir / \
                            f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(frame_filename), annotated_frame)
                        print(f"Saved frame: {frame_filename}")

                    # Simple exit mechanism when no GUI
                    if frame_count > 300:  # Stop after ~10 seconds
                        print("Stopping webcam detection (no GUI available)")
                        break

                frame_count += 1

        except KeyboardInterrupt:
            print("\nWebcam detection interrupted by user")
        finally:
            cap.release()
            if gui_available:
                cv2.destroyAllWindows()

        print("Webcam detection stopped.")

    def annotate_image(self, image: np.ndarray, results) -> np.ndarray:
        """
        Annotate image with detection results

        Args:
            image: Input image as numpy array
            results: YOLO detection results

        Returns:
            Annotated image
        """
        if results.boxes is None:
            return image

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)

            # Get class name and color
            class_name = self.class_names[class_id] if class_id < len(
                self.class_names) else f"Class_{class_id}"
            color = self.colors.get(class_name, (128, 128, 128))

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Create label
            label = f"{class_name}: {conf:.2f}"

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # Draw label background
            cv2.rectangle(image, (x1, y1 - text_height - baseline - 5),
                          (x1 + text_width, y1), color, -1)

            # Draw label text
            cv2.putText(image, label, (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return image

    def get_detection_stats(self, image_path: str, conf_threshold: float = 0.5):
        """
        Get detection statistics for an image

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections

        Returns:
            Dictionary with detection statistics
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)

        # Count detections by class
        stats = {class_name: 0 for class_name in self.class_names}
        total_detections = 0

        if results[0].boxes is not None:
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for class_id, conf in zip(class_ids, confidences):
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    stats[class_name] += 1
                    total_detections += 1

        stats['total'] = total_detections
        return stats


def main():
    parser = argparse.ArgumentParser(
        description='Face mask detection inference')
    parser.add_argument('--model', type=str, default='models/best.pt',
                        help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/dataset.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--source', type=str, required=True,
                        help='Input source (image path, video path, or camera ID)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save output')
    parser.add_argument('--show_stats', action='store_true',
                        help='Show detection statistics')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images (save only)')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        print("Please train a model first using: python src/train.py")
        return

    # Initialize detector
    detector = FaceMaskDetector(args.model, args.config)

    # Determine source type and run detection
    source = args.source

    if source.isdigit():
        # Webcam
        camera_id = int(source)
        detector.detect_webcam(camera_id, args.conf)

    elif os.path.isfile(source):
        # Check if it's an image or video
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

        file_ext = Path(source).suffix.lower()

        if file_ext in image_extensions:
            # Image
            print(f"Processing image: {source}")
            annotated_image = detector.detect_image(
                source, args.conf, args.save)

            if args.show_stats:
                stats = detector.get_detection_stats(source, args.conf)
                print(f"Detection statistics: {stats}")

            # Try to display image, fallback to saving if display fails
            if not args.no_display:
                try:
                    cv2.imshow('Face Mask Detection', annotated_image)
                    print("Press any key to close the image window...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print(f"Cannot display image (GUI not available): {e}")
                    print("Saving image instead...")

                    # Generate output filename if not provided
                    if args.save is None:
                        input_path = Path(source)
                        output_filename = f"{input_path.stem}_detected{input_path.suffix}"
                        output_path = input_path.parent / output_filename
                    else:
                        output_path = args.save

                    cv2.imwrite(str(output_path), annotated_image)
                    print(f"Annotated image saved to: {output_path}")
            else:
                # Save only mode
                if args.save is None:
                    input_path = Path(source)
                    output_filename = f"{input_path.stem}_detected{input_path.suffix}"
                    output_path = input_path.parent / output_filename
                else:
                    output_path = args.save

                cv2.imwrite(str(output_path), annotated_image)
                print(f"Annotated image saved to: {output_path}")

        elif file_ext in video_extensions:
            # Video
            print(f"Processing video: {source}")
            detector.detect_video(source, args.conf, args.save)

        else:
            print(f"Unsupported file format: {file_ext}")

    else:
        print(f"Source not found: {source}")


if __name__ == "__main__":
    main()
