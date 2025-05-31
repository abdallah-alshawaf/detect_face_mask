"""
Training script for face mask detection using YOLOv5
"""

import os
import argparse
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

from utils import load_config, validate_dataset


class FaceMaskTrainer:
    def __init__(self, config_path: str):
        """
        Initialize face mask trainer

        Args:
            config_path: Path to dataset configuration file
        """
        self.config = load_config(config_path)
        self.config_path = config_path

    def train(self,
              model_size: str = 'yolov8n',
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              device: str = 'auto',
              workers: int = 8,
              patience: int = 50,
              save_period: int = 10,
              project: str = 'runs/train',
              name: str = 'face_mask_detection',
              resume: bool = False,
              pretrained: bool = True):
        """
        Train the face mask detection model

        Args:
            model_size: YOLOv8 model size ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            device: Device to use for training ('auto', 'cpu', 'cuda', '0', '1', etc.)
            workers: Number of data loader workers
            patience: Early stopping patience
            save_period: Save model every N epochs
            project: Project directory
            name: Experiment name
            resume: Resume training from last checkpoint
            pretrained: Use pretrained weights
        """
        print("Starting face mask detection training...")
        print(f"Model: {model_size}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print(f"Device: {device}")

        # Validate dataset
        data_dir = self.config['path']
        if not validate_dataset(data_dir, self.config_path):
            raise ValueError("Dataset validation failed!")

        # Initialize model
        if pretrained:
            model = YOLO(f'{model_size}.pt')  # Load pretrained model
        else:
            model = YOLO(f'{model_size}.yaml')  # Load model architecture only

        # Configure training parameters
        train_args = {
            'data': self.config_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'device': device,
            'workers': workers,
            'patience': patience,
            'save_period': save_period,
            'project': project,
            'name': name,
            'exist_ok': True,
            'pretrained': pretrained,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': resume,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }

        # Start training
        try:
            results = model.train(**train_args)

            print("\nTraining completed successfully!")
            print(f"Best model saved to: {results.save_dir}")

            # Print training results
            if hasattr(results, 'results_dict'):
                print("\nTraining Results:")
                for key, value in results.results_dict.items():
                    print(f"{key}: {value}")

            return results

        except Exception as e:
            print(f"Training failed: {e}")
            raise

    def validate_model(self, model_path: str, data_path: str = None):
        """
        Validate trained model

        Args:
            model_path: Path to trained model
            data_path: Path to dataset config (optional)
        """
        print(f"Validating model: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        model = YOLO(model_path)

        # Use dataset config if provided, otherwise use the one from initialization
        config_path = data_path if data_path else self.config_path

        # Run validation
        results = model.val(data=config_path, verbose=True)

        print("\nValidation Results:")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")

        # Print per-class results
        if hasattr(results.box, 'ap_class_index') and hasattr(results.box, 'ap'):
            class_names = self.config['names']
            print("\nPer-class mAP@0.5:")
            for i, class_idx in enumerate(results.box.ap_class_index):
                if i < len(results.box.ap50):
                    print(
                        f"{class_names[class_idx]}: {results.box.ap50[i]:.4f}")

        return results

    def export_model(self, model_path: str, format: str = 'onnx'):
        """
        Export trained model to different formats

        Args:
            model_path: Path to trained model
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
        """
        print(f"Exporting model to {format} format...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        model = YOLO(model_path)

        # Export model
        exported_path = model.export(format=format)

        print(f"Model exported to: {exported_path}")
        return exported_path


def main():
    parser = argparse.ArgumentParser(
        description='Train face mask detection model')
    parser.add_argument('--config', type=str, default='config/dataset.yaml',
                        help='Path to dataset configuration file')
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s',
                                 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for training')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loader workers')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='face_mask_detection',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='Do not use pretrained weights')
    parser.add_argument('--validate_only', type=str, default=None,
                        help='Only validate the specified model')
    parser.add_argument('--export', type=str, default=None,
                        help='Export model to specified format')
    parser.add_argument('--export_format', type=str, default='onnx',
                        help='Export format')

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' does not exist")
        return

    # Initialize trainer
    trainer = FaceMaskTrainer(args.config)

    if args.validate_only:
        # Only validate model
        trainer.validate_model(args.validate_only)
    elif args.export:
        # Only export model
        trainer.export_model(args.export, args.export_format)
    else:
        # Train model
        results = trainer.train(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            project=args.project,
            name=args.name,
            resume=args.resume,
            pretrained=not args.no_pretrained
        )

        # Validate the trained model
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            print("\nValidating best model...")
            trainer.validate_model(best_model_path)

            # Copy best model to models directory
            models_dir = 'models'
            os.makedirs(models_dir, exist_ok=True)
            import shutil
            shutil.copy2(best_model_path, os.path.join(models_dir, 'best.pt'))
            print(
                f"Best model copied to: {os.path.join(models_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
