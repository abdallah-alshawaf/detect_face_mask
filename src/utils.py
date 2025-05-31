"""
Utility functions for face mask detection project
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import random
from typing import List, Tuple, Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_xml_annotation(xml_path: str) -> Dict[str, Any]:
    """
    Parse PASCAL VOC XML annotation file

    Args:
        xml_path: Path to XML annotation file

    Returns:
        Dictionary containing image info and annotations
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image information
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Get object annotations
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')

        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })

    return {
        'filename': filename,
        'width': width,
        'height': height,
        'objects': objects
    }


def convert_bbox_to_yolo(bbox: List[int], img_width: int, img_height: int) -> List[float]:
    """
    Convert PASCAL VOC bbox to YOLO format

    Args:
        bbox: [xmin, ymin, xmax, ymax]
        img_width: Image width
        img_height: Image height

    Returns:
        [x_center, y_center, width, height] normalized to [0, 1]
    """
    xmin, ymin, xmax, ymax = bbox

    # Calculate center coordinates and dimensions
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin

    # Normalize to [0, 1]
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return [x_center, y_center, width, height]


def convert_yolo_to_bbox(yolo_coords: List[float], img_width: int, img_height: int) -> List[int]:
    """
    Convert YOLO format to PASCAL VOC bbox

    Args:
        yolo_coords: [x_center, y_center, width, height] normalized
        img_width: Image width
        img_height: Image height

    Returns:
        [xmin, ymin, xmax, ymax]
    """
    x_center, y_center, width, height = yolo_coords

    # Denormalize
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Calculate corners
    xmin = int(x_center - width / 2)
    ymin = int(y_center - height / 2)
    xmax = int(x_center + width / 2)
    ymax = int(y_center + height / 2)

    return [xmin, ymin, xmax, ymax]


def visualize_annotations(image_path: str, annotation_path: str, class_names: List[str]):
    """
    Visualize image with bounding box annotations

    Args:
        image_path: Path to image file
        annotation_path: Path to XML annotation file
        class_names: List of class names
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Parse annotations
    annotation = parse_xml_annotation(annotation_path)

    # Define colors for each class
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

    # Draw bounding boxes
    for obj in annotation['objects']:
        bbox = obj['bbox']
        class_name = obj['name']

        # Get class index and color
        try:
            class_idx = class_names.index(class_name)
            color = colors[class_idx % len(colors)]
        except ValueError:
            color = (128, 128, 128)  # Gray for unknown classes

        # Draw rectangle
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Add label
        label = f"{class_name}"
        cv2.putText(image, label, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display image
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f"Image: {annotation['filename']}")
    plt.axis('off')
    plt.show()


def create_train_val_split(image_dir: str, train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Create train/validation split from image directory

    Args:
        image_dir: Directory containing images
        train_ratio: Ratio of images for training

    Returns:
        Tuple of (train_files, val_files)
    """
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    # Convert to strings and sort
    image_files = [str(f) for f in image_files]
    image_files.sort()

    # Shuffle for random split
    random.shuffle(image_files)

    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    return train_files, val_files


def calculate_class_distribution(annotation_dir: str) -> Dict[str, int]:
    """
    Calculate class distribution from annotation files

    Args:
        annotation_dir: Directory containing XML annotation files

    Returns:
        Dictionary with class counts
    """
    class_counts = {}

    # Process all XML files
    for xml_file in Path(annotation_dir).glob("*.xml"):
        annotation = parse_xml_annotation(str(xml_file))

        for obj in annotation['objects']:
            class_name = obj['name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return class_counts


def plot_class_distribution(class_counts: Dict[str, int]):
    """
    Plot class distribution as bar chart

    Args:
        class_counts: Dictionary with class counts
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=['red', 'green', 'blue'])
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Classes')
    plt.ylabel('Number of Instances')
    plt.xticks(rotation=45)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def ensure_dir(directory: str):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_image_size(image_path: str) -> Tuple[int, int]:
    """
    Get image dimensions

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height)
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    return width, height


def validate_dataset(data_dir: str, config_path: str) -> bool:
    """
    Validate dataset structure and files

    Args:
        data_dir: Path to data directory
        config_path: Path to dataset configuration file

    Returns:
        True if dataset is valid, False otherwise
    """
    try:
        # Load config
        config = load_config(config_path)

        # Check required directories
        required_dirs = ['train', 'val']
        for dir_name in required_dirs:
            dir_path = os.path.join(data_dir, dir_name)
            if not os.path.exists(dir_path):
                print(f"Missing directory: {dir_path}")
                return False

        # Check if there are images and labels
        for dir_name in required_dirs:
            images_dir = os.path.join(data_dir, dir_name, 'images')
            labels_dir = os.path.join(data_dir, dir_name, 'labels')

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                print(f"Missing images or labels directory in {dir_name}")
                return False

            # Count files
            image_files = list(Path(images_dir).glob("*.jpg")) + \
                list(Path(images_dir).glob("*.png"))
            label_files = list(Path(labels_dir).glob("*.txt"))

            if len(image_files) == 0:
                print(f"No images found in {images_dir}")
                return False

            if len(label_files) == 0:
                print(f"No labels found in {labels_dir}")
                return False

            print(f"{dir_name}: {len(image_files)} images, {len(label_files)} labels")

        print("Dataset validation passed!")
        return True

    except Exception as e:
        print(f"Dataset validation failed: {e}")
        return False
