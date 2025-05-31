"""
Data preparation script for face mask detection
Converts PASCAL VOC XML annotations to YOLO format and organizes dataset
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import random

from utils import (
    parse_xml_annotation,
    convert_bbox_to_yolo,
    create_train_val_split,
    calculate_class_distribution,
    plot_class_distribution,
    ensure_dir,
    load_config
)


class DataPreparator:
    def __init__(self, config_path: str):
        """
        Initialize data preparator

        Args:
            config_path: Path to dataset configuration file
        """
        self.config = load_config(config_path)
        self.class_names = self.config['names']
        self.class_to_id = {name: idx for idx,
                            name in enumerate(self.class_names)}

    def convert_xml_to_yolo(self, xml_path: str, img_width: int, img_height: int) -> str:
        """
        Convert XML annotation to YOLO format

        Args:
            xml_path: Path to XML annotation file
            img_width: Image width
            img_height: Image height

        Returns:
            YOLO format annotation string
        """
        annotation = parse_xml_annotation(xml_path)
        yolo_lines = []

        for obj in annotation['objects']:
            class_name = obj['name']
            bbox = obj['bbox']

            # Skip if class not in our class list
            if class_name not in self.class_to_id:
                print(f"Warning: Unknown class '{class_name}' in {xml_path}")
                continue

            class_id = self.class_to_id[class_name]
            yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)

            # Format: class_id x_center y_center width height
            line = f"{class_id} {' '.join(map(str, yolo_bbox))}"
            yolo_lines.append(line)

        return '\n'.join(yolo_lines)

    def prepare_dataset(self,
                        images_dir: str,
                        annotations_dir: str,
                        output_dir: str,
                        train_ratio: float = 0.8,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.05):
        """
        Prepare dataset by converting annotations and organizing files

        Args:
            images_dir: Directory containing original images
            annotations_dir: Directory containing XML annotations
            output_dir: Output directory for organized dataset
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
        """
        print("Starting dataset preparation...")

        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                "Train, validation, and test ratios must sum to 1.0")

        # Create output directories
        splits = ['train', 'val', 'test']
        for split in splits:
            ensure_dir(os.path.join(output_dir, split, 'images'))
            ensure_dir(os.path.join(output_dir, split, 'labels'))

        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f"*{ext}"))
            image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))

        # Convert to list of strings and sort
        image_files = [str(f) for f in image_files]
        image_files.sort()

        print(f"Found {len(image_files)} images")

        # Filter images that have corresponding annotations
        valid_images = []
        for img_path in image_files:
            img_name = Path(img_path).stem
            xml_path = os.path.join(annotations_dir, f"{img_name}.xml")

            if os.path.exists(xml_path):
                valid_images.append(img_path)
            else:
                print(f"Warning: No annotation found for {img_path}")

        print(f"Found {len(valid_images)} images with annotations")

        if len(valid_images) == 0:
            raise ValueError("No valid image-annotation pairs found!")

        # Shuffle for random split
        random.shuffle(valid_images)

        # Calculate split indices
        n_total = len(valid_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        # Split the data
        train_images = valid_images[:n_train]
        val_images = valid_images[n_train:n_train + n_val]
        test_images = valid_images[n_train + n_val:]

        print(
            f"Dataset split: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

        # Process each split
        splits_data = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

        total_processed = 0
        class_counts = {}

        for split_name, split_images in splits_data.items():
            if len(split_images) == 0:
                continue

            print(f"\nProcessing {split_name} split...")

            split_output_dir = os.path.join(output_dir, split_name)
            images_output_dir = os.path.join(split_output_dir, 'images')
            labels_output_dir = os.path.join(split_output_dir, 'labels')

            for img_path in tqdm(split_images, desc=f"Processing {split_name}"):
                img_name = Path(img_path).stem
                img_ext = Path(img_path).suffix
                xml_path = os.path.join(annotations_dir, f"{img_name}.xml")

                try:
                    # Parse annotation to get image dimensions
                    annotation = parse_xml_annotation(xml_path)
                    img_width = annotation['width']
                    img_height = annotation['height']

                    # Convert to YOLO format
                    yolo_annotation = self.convert_xml_to_yolo(
                        xml_path, img_width, img_height)

                    # Skip if no valid objects
                    if not yolo_annotation.strip():
                        print(f"Warning: No valid objects in {xml_path}")
                        continue

                    # Copy image
                    dst_img_path = os.path.join(
                        images_output_dir, f"{img_name}{img_ext}")
                    shutil.copy2(img_path, dst_img_path)

                    # Save YOLO annotation
                    dst_label_path = os.path.join(
                        labels_output_dir, f"{img_name}.txt")
                    with open(dst_label_path, 'w') as f:
                        f.write(yolo_annotation)

                    # Count classes
                    for obj in annotation['objects']:
                        class_name = obj['name']
                        if class_name in self.class_to_id:
                            class_counts[class_name] = class_counts.get(
                                class_name, 0) + 1

                    total_processed += 1

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

        print(f"\nDataset preparation completed!")
        print(f"Total processed: {total_processed} images")
        print(f"Class distribution: {class_counts}")

        # Plot class distribution
        if class_counts:
            plot_class_distribution(class_counts)

        return class_counts

    def create_sample_dataset(self,
                              images_dir: str,
                              annotations_dir: str,
                              output_dir: str,
                              n_samples: int = 50):
        """
        Create a small sample dataset for testing

        Args:
            images_dir: Directory containing original images
            annotations_dir: Directory containing XML annotations
            output_dir: Output directory for sample dataset
            n_samples: Number of samples to include
        """
        print(f"Creating sample dataset with {n_samples} images...")

        # Get all valid image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []

        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f"*{ext}"))
            image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))

        # Filter images that have corresponding annotations
        valid_images = []
        for img_path in image_files:
            img_name = Path(img_path).stem
            xml_path = os.path.join(annotations_dir, f"{img_name}.xml")

            if os.path.exists(xml_path):
                valid_images.append(str(img_path))

        # Randomly sample
        random.shuffle(valid_images)
        sample_images = valid_images[:min(n_samples, len(valid_images))]

        # Prepare sample dataset
        self.prepare_dataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )

        print(f"Sample dataset created with {len(sample_images)} images")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare face mask detection dataset')
    parser.add_argument('--images_dir', type=str, default='data/images',
                        help='Directory containing original images')
    parser.add_argument('--annotations_dir', type=str, default='data/annotations',
                        help='Directory containing XML annotations')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for organized dataset')
    parser.add_argument('--config', type=str, default='config/dataset.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data for training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of data for validation')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='Ratio of data for testing')
    parser.add_argument('--sample', action='store_true',
                        help='Create a small sample dataset for testing')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples for sample dataset')

    args = parser.parse_args()

    # Validate input directories
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory '{args.images_dir}' does not exist")
        print("Please download the dataset from Kaggle and extract it to the data/ folder")
        return

    if not os.path.exists(args.annotations_dir):
        print(
            f"Error: Annotations directory '{args.annotations_dir}' does not exist")
        print("Please download the dataset from Kaggle and extract it to the data/ folder")
        return

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        return

    # Initialize data preparator
    preparator = DataPreparator(args.config)

    # Analyze dataset first
    print("Analyzing dataset...")
    class_counts = calculate_class_distribution(args.annotations_dir)
    print(f"Found classes: {class_counts}")

    if args.sample:
        # Create sample dataset
        preparator.create_sample_dataset(
            images_dir=args.images_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir,
            n_samples=args.n_samples
        )
    else:
        # Prepare full dataset
        preparator.prepare_dataset(
            images_dir=args.images_dir,
            annotations_dir=args.annotations_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )


if __name__ == "__main__":
    main()
