"""
Quick Start Script for Face Mask Detection
Provides an easy way to test the system without the full dataset
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print welcome banner"""
    print("="*60)
    print("🎭 AI FACE MASK DETECTOR - QUICK START")
    print("="*60)
    print()
    print("Welcome to the AI Face Mask Detection system!")
    print("This quick start will help you test the system immediately.")
    print()


def check_setup():
    """Check if basic setup is complete"""
    print("Checking setup...")

    # Check if requirements are installed
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        print("✓ Dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Please run: python setup.py")
        return False


def run_demo():
    """Run the demonstration"""
    print("\nStarting demonstration...")
    print("This will show you how the face mask detection works.")
    print()

    # Ask user what they want to do
    print("Choose an option:")
    print("1. View sample images with simulated detection")
    print("2. Test with webcam (if available)")
    print("3. Exit")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            print("\nRunning sample image demo...")
            try:
                subprocess.run([sys.executable, "demo.py",
                               "--mode", "samples"], check=True)
                break
            except subprocess.CalledProcessError:
                print("Demo failed. Please check your setup.")
                break

        elif choice == '2':
            print("\nStarting webcam demo...")
            print("Note: This requires a working webcam.")
            print("Press 'q' in the video window to quit.")
            try:
                subprocess.run([sys.executable, "demo.py",
                               "--mode", "webcam"], check=True)
                break
            except subprocess.CalledProcessError:
                print("Webcam demo failed. Please check your camera.")
                break

        elif choice == '3':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def show_next_steps():
    """Show what to do next"""
    print("\n" + "="*60)
    print("🚀 NEXT STEPS FOR FULL FUNCTIONALITY")
    print("="*60)
    print()
    print("What you just saw was a demonstration using a general object detection model.")
    print("For actual face mask detection, follow these steps:")
    print()
    print("1. 📥 Download the dataset:")
    print("   • Go to: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection")
    print("   • Download and extract to data/ folder")
    print()
    print("2. 🔧 Prepare the data:")
    print("   python src/data_preparation.py")
    print()
    print("3. 🎯 Train the model:")
    print("   python src/train.py")
    print()
    print("4. 🔍 Run real detection:")
    print("   python src/detect.py --source path/to/image.jpg")
    print("   python src/detect.py --source 0  # for webcam")
    print()
    print("5. 📊 Model will detect:")
    print("   • ✅ People wearing masks correctly")
    print("   • ❌ People not wearing masks")
    print("   • ⚠️  People wearing masks incorrectly")
    print()
    print("For detailed instructions, see README.md")
    print("="*60)


def main():
    """Main function"""
    print_banner()

    # Check if setup is complete
    if not check_setup():
        print("\nPlease run the setup first:")
        print("python setup.py")
        return

    # Run demonstration
    run_demo()

    # Show next steps
    show_next_steps()

    print("\nThank you for trying the AI Face Mask Detector! 🎭")


if __name__ == "__main__":
    main()
