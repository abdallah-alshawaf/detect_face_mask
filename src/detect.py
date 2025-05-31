"""
Enhanced detection script for face mask detection
Supports inference on images, videos, and webcam
Includes automatic image format conversion to PNG
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import time
from ultralytics import YOLO
import torch
from PIL import Image

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

        # Supported image formats
        self.supported_formats = [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            '.webp', '.gif', '.jfif', '.ppm', '.pgm', '.pbm'
        ]

        # Create converted images directory
        self.converted_dir = Path("converted_images")
        self.converted_dir.mkdir(exist_ok=True)

        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Supported formats: {', '.join(self.supported_formats)}")

    def convert_to_png(self, image_path: str, output_dir: str = None) -> str:
        """
        Convert any image format to PNG

        Args:
            image_path: Path to input image
            output_dir: Directory to save converted PNG (optional)

        Returns:
            Path to converted PNG file
        """
        input_path = Path(image_path)

        # Generate output path
        if output_dir:
            output_path = Path(output_dir) / f"{input_path.stem}.png"
        else:
            output_path = self.converted_dir / f"{input_path.stem}.png"

        # If already PNG and in same location, just return the path
        if input_path.suffix.lower() == '.png' and not output_dir:
            return str(input_path)

        try:
            # Try with PIL first (handles more formats)
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for formats like RGBA, P, etc.)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    if len(img.split()) > 3:  # Has alpha channel
                        rgb_img.paste(img, mask=img.split()[-1])
                    else:
                        rgb_img.paste(img)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                img.save(output_path, 'PNG')
                print(f"Converted {input_path.name} -> {output_path.name}")
                return str(output_path)

        except Exception as e:
            # Fallback to OpenCV
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(
                        f"Could not load image with PIL or OpenCV: {image_path}")

                cv2.imwrite(str(output_path), image)
                print(
                    f"Converted {input_path.name} -> {output_path.name} (via OpenCV)")
                return str(output_path)

            except Exception as e2:
                raise ValueError(f"Failed to convert image: {e}, {e2}")

    def load_and_convert_image(self, image_path: str, auto_convert: bool = True):
        """
        Load image and optionally convert to PNG format

        Args:
            image_path: Path to input image
            auto_convert: Whether to automatically convert to PNG

        Returns:
            tuple: (image_array, converted_path)
        """
        input_path = Path(image_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Check if format is supported
        if input_path.suffix.lower() not in self.supported_formats:
            print(
                f"Warning: Unsupported format {input_path.suffix}, attempting to process anyway...")

        converted_path = image_path

        # Convert to PNG if requested and not already PNG
        if auto_convert and input_path.suffix.lower() != '.png':
            try:
                converted_path = self.convert_to_png(image_path)
            except Exception as e:
                print(f"Conversion failed: {e}. Using original file.")
                converted_path = image_path

        # Load image with OpenCV
        image = cv2.imread(converted_path)
        if image is None:
            raise ValueError(f"Could not load image: {converted_path}")

        return image, converted_path

    def detect_image(self, image_path: str, conf_threshold: float = 0.5, save_path: str = None, auto_convert: bool = True):
        """
        Detect face masks in a single image with automatic format conversion

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            save_path: Path to save output image (optional)
            auto_convert: Whether to automatically convert input to PNG

        Returns:
            Annotated image as numpy array
        """
        print(f"Processing image: {Path(image_path).name}")

        # Load and optionally convert image
        image, converted_path = self.load_and_convert_image(
            image_path, auto_convert)

        if converted_path != image_path:
            print(f"Using converted image: {Path(converted_path).name}")

        # Run inference
        results = self.model(image, conf=conf_threshold, verbose=False)

        # Annotate image
        annotated_image = self.annotate_image(image.copy(), results[0])

        # Save if path provided (always save as PNG for consistency)
        if save_path:
            save_path_png = Path(save_path)
            if save_path_png.suffix.lower() != '.png':
                save_path_png = save_path_png.with_suffix('.png')
            cv2.imwrite(str(save_path_png), annotated_image)
            print(f"Output saved to: {save_path_png}")

        return annotated_image

    def batch_convert_images(self, input_dir: str, output_dir: str = None):
        """
        Convert all images in a directory to PNG format

        Args:
            input_dir: Directory containing images to convert
            output_dir: Output directory for PNG files (optional)
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        else:
            output_path = self.converted_dir

        print(f"Converting images from {input_path} to {output_path}")

        converted_count = 0
        failed_count = 0

        # Find all image files
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        print(f"Found {len(image_files)} image files")

        for image_file in image_files:
            try:
                self.convert_to_png(str(image_file), str(output_path))
                converted_count += 1
            except Exception as e:
                print(f"Failed to convert {image_file.name}: {e}")
                failed_count += 1

        print(
            f"Batch conversion complete: {converted_count} converted, {failed_count} failed")
        return converted_count, failed_count

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

                # Display frame with error handling
                try:
                    cv2.imshow('Face Mask Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("GUI display not available, processing without display...")

                frame_count += 1

                # Print progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    print(
                        f"Progress: {frame_count}/{total_frames} frames ({fps_current:.1f} FPS)")

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
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except cv2.error as e:
                        print(f"GUI display failed: {e}")
                        print("Switching to save mode...")
                        gui_available = False
                        save_dir = Path("webcam_frames")
                        save_dir.mkdir(exist_ok=True)

                # Save frames if no GUI
                if not gui_available:
                    if frame_count % 30 == 0:  # Save every second
                        frame_filename = save_dir / \
                            f"frame_{frame_count:06d}.png"
                        cv2.imwrite(str(frame_filename), annotated_frame)
                        print(f"Saved: {frame_filename}")

                    if frame_count > 300:  # Stop after 10 seconds
                        break

                frame_count += 1

        except KeyboardInterrupt:
            print("\nWebcam detection stopped by user")
        finally:
            cap.release()
            if gui_available:
                cv2.destroyAllWindows()

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
        # Load and convert image
        image, converted_path = self.load_and_convert_image(image_path)

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
        description='Enhanced face mask detection with automatic format conversion')
    parser.add_argument('--model', type=str,
                        default='models/best.pt', help='Path to trained model')
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
    parser.add_argument('--no_convert', action='store_true',
                        help='Disable automatic PNG conversion')
    parser.add_argument('--batch_convert', type=str, default=None,
                        help='Convert all images in directory to PNG')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' does not exist")
        print("Please train a model first using: python src/train.py")
        return

    # Initialize detector
    detector = FaceMaskDetector(args.model, args.config)

    # Batch conversion mode
    if args.batch_convert:
        detector.batch_convert_images(args.batch_convert)
        return

    # Determine source type and run detection
    source = args.source

    if source.isdigit():
        # Webcam
        camera_id = int(source)
        detector.detect_webcam(camera_id, args.conf)

    elif os.path.isfile(source):
        # Check if it's an image or video
        image_extensions = detector.supported_formats
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']

        file_ext = Path(source).suffix.lower()

        if file_ext in image_extensions:
            # Image
            annotated_image = detector.detect_image(
                source, args.conf, args.save, auto_convert=not args.no_convert)

            if args.show_stats:
                stats = detector.get_detection_stats(source, args.conf)
                print(f"Detection statistics: {stats}")

            # Try to display image, fallback to saving if display fails
            if not args.no_display:
                try:
                    # Get screen dimensions (approximate)
                    try:
                        import tkinter as tk
                        root = tk.Tk()
                        screen_width = root.winfo_screenwidth()
                        screen_height = root.winfo_screenheight()
                        root.destroy()
                    except:
                        # Fallback dimensions if tkinter fails
                        screen_width, screen_height = 1920, 1080

                    # Calculate max display size (80% of screen)
                    max_width = int(screen_width * 0.8)
                    max_height = int(screen_height * 0.8)

                    # Get image dimensions
                    img_height, img_width = annotated_image.shape[:2]

                    # Calculate scaling factor to fit within screen
                    scale_w = max_width / img_width
                    scale_h = max_height / img_height
                    scale = min(scale_w, scale_h, 1.0)  # Don't upscale

                    # Resize image for display if needed
                    if scale < 1.0:
                        display_width = int(img_width * scale)
                        display_height = int(img_height * scale)
                        display_image = cv2.resize(
                            annotated_image, (display_width, display_height))
                        print(
                            f"Resized for display: {img_width}x{img_height} -> {display_width}x{display_height}")
                    else:
                        display_image = annotated_image

                    # Create named window with resizable property
                    window_name = 'Face Mask Detection'
                    cv2.namedWindow(
                        window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

                    # Set window to be resizable
                    cv2.resizeWindow(
                        window_name, display_image.shape[1], display_image.shape[0])

                    # Display the image
                    cv2.imshow(window_name, display_image)
                    print("Press any key to close the image window...")
                    print("You can resize the window by dragging the corners!")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as e:
                    print(f"Cannot display image (GUI not available): {e}")
                    print("Saving image instead...")

                    if args.save is None:
                        input_path = Path(source)
                        output_filename = f"{input_path.stem}_detected.png"
                        output_path = input_path.parent / output_filename
                    else:
                        output_path = Path(args.save).with_suffix('.png')

                    cv2.imwrite(str(output_path), annotated_image)
                    print(f"Annotated image saved to: {output_path}")
                except Exception as e:
                    print(f"Display error: {e}")
                    print("Saving image instead...")

                    if args.save is None:
                        input_path = Path(source)
                        output_filename = f"{input_path.stem}_detected.png"
                        output_path = input_path.parent / output_filename
                    else:
                        output_path = Path(args.save).with_suffix('.png')

                    cv2.imwrite(str(output_path), annotated_image)
                    print(f"Annotated image saved to: {output_path}")
            else:
                # Save only mode
                if args.save is None:
                    input_path = Path(source)
                    output_filename = f"{input_path.stem}_detected.png"
                    output_path = input_path.parent / output_filename
                else:
                    output_path = Path(args.save).with_suffix('.png')

                cv2.imwrite(str(output_path), annotated_image)
                print(f"Annotated image saved to: {output_path}")

        elif file_ext in video_extensions:
            # Video
            print(f"Processing video: {source}")
            detector.detect_video(source, args.conf, args.save)

        else:
            print(f"Unsupported file format: {file_ext}")
            print(f"Supported image formats: {', '.join(image_extensions)}")
            print(f"Supported video formats: {', '.join(video_extensions)}")

    else:
        print(f"Source not found: {source}")


if __name__ == "__main__":
    main()
