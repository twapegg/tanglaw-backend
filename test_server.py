#!/usr/bin/env python3
"""
Test script for the Image Processing Pipeline Server
"""

import requests
import json
import base64
from PIL import Image
import io
import argparse


def test_health(base_url):
    """Test the health endpoint."""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{base_url}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_process_image(base_url, image_path, save_output=False):
    """Test the process_image endpoint."""
    print("\n=== Testing Process Image Endpoint ===")
    print(f"Uploading: {image_path}")

    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(f"{base_url}/process_image", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Processing Time: {data['processing_time']}s")
        print(f"Total Images Generated: {data['total_images']}")

        print("\nProcessed Images:")
        for i, img_data in enumerate(data["images"], 1):
            print(f"  {i}. Type: {img_data['type']}")
            print(f"     Enhancement: {img_data['enhancement']}")
            print(f"     Darkening: {img_data['darkening_level']}%")
            print(f"     Faces Detected: {img_data['faces_detected']}")

            if save_output:
                # Save decoded image
                img_bytes = base64.b64decode(img_data["image"])
                img = Image.open(io.BytesIO(img_bytes))
                output_name = f"output_{img_data['type']}.jpg"
                img.save(output_name)
                print(f"     Saved to: {output_name}")

        return True
    else:
        print(f"Error: {response.json()}")
        return False


def test_process_image_files(base_url, image_path, output_zip="output.zip"):
    """Test the process_image_files endpoint."""
    print("\n=== Testing Process Image Files Endpoint ===")
    print(f"Uploading: {image_path}")

    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(f"{base_url}/process_image_files", files=files)

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        with open(output_zip, "wb") as f:
            f.write(response.content)
        print(f"ZIP file saved to: {output_zip}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Image Processing Pipeline Server"
    )
    parser.add_argument("--url", default="http://localhost:5000", help="Server URL")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--save", action="store_true", help="Save output images")
    parser.add_argument(
        "--files", action="store_true", help="Test file download endpoint"
    )

    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Test health endpoint
    if not test_health(base_url):
        print("Health check failed!")
        return

    # Test process_image endpoint
    if args.files:
        test_process_image_files(base_url, args.image)
    else:
        test_process_image(base_url, args.image, args.save)


if __name__ == "__main__":
    main()
