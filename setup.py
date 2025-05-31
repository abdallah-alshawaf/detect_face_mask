"""
Setup script for Face Mask Detection project
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(
            f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(
            f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or higher")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")

    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False

    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False

    return True


def create_directories():
    """Create necessary directories"""
    print("\nCreating project directories...")

    directories = [
        'data/images',
        'data/annotations',
        'data/train/images',
        'data/train/labels',
        'data/val/images',
        'data/val/labels',
        'data/test/images',
        'data/test/labels',
        'models',
        'runs',
        'sample_images'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

    return True


def download_sample_data():
    """Download sample data for testing"""
    print("\nDownloading sample data...")

    try:
        # This would download sample images in a real implementation
        # For now, we'll just create placeholder files
        sample_dir = Path('sample_images')
        sample_dir.mkdir(exist_ok=True)

        # Create a simple test image
        import cv2
        import numpy as np

        # Create a simple test image
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        img[:] = (100, 100, 100)  # Gray background

        # Add text
        cv2.putText(img, "Sample Image for Testing", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save image
        cv2.imwrite('sample_images/test_image.jpg', img)
        print("✓ Created sample test image")

        return True

    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
        return False


def check_gpu_support():
    """Check if GPU support is available"""
    print("\nChecking GPU support...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU support available: {gpu_count} GPU(s)")
            print(f"  Primary GPU: {gpu_name}")
        else:
            print("⚠ No GPU support detected. Training will use CPU (slower)")
        return True
    except ImportError:
        print(
            "⚠ PyTorch not installed yet. GPU check will be performed after installation.")
        return True


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Download the dataset:")
    print("   - Go to: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
    print("   - Download and extract to data/ folder")
    print("   - Place images in data/images/")
    print("   - Place annotations in data/annotations/")

    print("\n2. Prepare the dataset:")
    print("   python src/data_preparation.py")

    print("\n3. Train the model:")
    print("   python src/train.py")

    print("\n4. Run detection:")
    print("   python src/detect.py --source path/to/image.jpg")
    print("   python src/detect.py --source 0  # for webcam")

    print("\n5. Try the demo (works without dataset):")
    print("   python demo.py --mode samples")
    print("   python demo.py --mode webcam")

    print("\nFor more information, see README.md")
    print("="*60)


def main():
    """Main setup function"""
    print("Face Mask Detection - Setup Script")
    print("="*40)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check GPU support (before installing dependencies)
    check_gpu_support()

    # Install dependencies
    if not install_dependencies():
        print("\n✗ Failed to install dependencies")
        sys.exit(1)

    # Create directories
    if not create_directories():
        print("\n✗ Failed to create directories")
        sys.exit(1)

    # Download sample data
    if not download_sample_data():
        print("\n⚠ Failed to create sample data, but setup can continue")

    # Final GPU check (after installing PyTorch)
    check_gpu_support()

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
