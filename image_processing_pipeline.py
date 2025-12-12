#!/usr/bin/env python3
"""
Image Processing Pipeline for Face Detection
Combines darkening, enhancement (classical and deep learning), and face detection.

Usage:
    python image_processing_pipeline.py --input INPUT_FOLDER [OPTIONS]
"""

import argparse
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import time
import torch
import sys
import logging
from datetime import datetime


class ImageProcessingPipeline:
    """Main pipeline class for image processing and face detection."""

    def __init__(
        self,
        input_folder,
        output_base="output",
        batch_size=32,
        enable_darkening=True,
        enable_classical=True,
        enable_deep=True,
        enable_detection=True,
        max_display=5,
    ):
        """
        Initialize the pipeline.

        Args:
            input_folder: Path to input images
            output_base: Base directory for all outputs
            batch_size: Number of images to process in parallel
            enable_darkening: Enable darkening stage
            enable_classical: Enable classical enhancement
            enable_deep: Enable deep learning enhancement
            enable_detection: Enable face detection
            max_display: Maximum images to save with visualizations
        """
        self.input_folder = input_folder
        self.output_base = output_base
        self.preprocessed_folder = os.path.join(output_base, "preprocessed")
        self.results_folder = os.path.join(output_base, "results")
        self.batch_size = batch_size
        self.enable_darkening = enable_darkening
        self.enable_classical = enable_classical
        self.enable_deep = enable_deep
        self.enable_detection = enable_detection
        self.max_display = max_display

        # Setup logging
        self.setup_logging()

        # Create output directories
        os.makedirs(self.preprocessed_folder, exist_ok=True)
        os.makedirs(self.results_folder, exist_ok=True)

        # Initialize detector
        self.detector = None
        if self.enable_detection:
            try:
                self.detector = MTCNN()
                logging.info("MTCNN detector initialized")
            except Exception as e:
                logging.error(f"Failed to initialize MTCNN detector: {e}")
                logging.error(
                    "Please install tensorflow: pip install tensorflow>=2.12.0"
                )
                self.enable_detection = False

        # Initialize deep learning model
        self.model = None
        self.device = None
        if self.enable_deep:
            self.setup_deep_model()

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = os.path.join(self.output_base, "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        logging.info(f"Pipeline initialized. Log file: {log_file}")

    def setup_deep_model(self):
        """Setup Zero-DCE deep learning model."""
        try:
            # Check if Zero-DCE exists
            zero_dce_path = os.path.join(os.getcwd(), "Zero-DCE", "Zero-DCE_code")
            if not os.path.exists(zero_dce_path):
                logging.warning(
                    "Zero-DCE not found. Skipping deep learning enhancement."
                )
                self.enable_deep = False
                return

            if zero_dce_path not in sys.path:
                sys.path.append(zero_dce_path)

            from model import enhance_net_nopool

            model_path = os.path.join(zero_dce_path, "snapshots", "Epoch99.pth")
            if not os.path.exists(model_path):
                logging.warning(
                    f"Model file not found: {model_path}. Skipping deep learning enhancement."
                )
                self.enable_deep = False
                return

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = enhance_net_nopool().to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            logging.info(f"Zero-DCE model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load deep learning model: {e}")
            self.enable_deep = False

    def darken_batch(
        self, arr, percent, noise_amount=0.005, gamma=1.1, curve=1.4, bright_factor=0.3
    ):
        """Darken a batch of images."""
        arr = arr.astype("float32") / 255.0
        linear = arr**gamma
        linear_dark = linear * (1 - percent / 100.0)
        linear_dark = linear_dark**curve
        linear_dark = linear_dark * (1 - bright_factor * linear)

        noise = np.random.normal(0.0, noise_amount, linear_dark.shape)
        linear_noisy = np.clip(linear_dark + noise, 0.0, 1.0)

        srgb = np.clip(linear_noisy ** (1.0 / gamma), 0, 1)
        out = (srgb * 255).astype("uint8")
        return out

    def darken_images(self):
        """Stage 1: Darken all images at 50% and 80% levels."""
        if not self.enable_darkening:
            logging.info("Darkening stage skipped")
            return

        logging.info("=" * 60)
        logging.info("STAGE 1: DARKENING IMAGES")
        logging.info("=" * 60)

        start_time = time.time()

        filenames = [
            f
            for f in os.listdir(self.input_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not filenames:
            logging.warning(f"No images found in {self.input_folder}")
            return

        logging.info(f"Found {len(filenames)} images to darken")

        for start in range(0, len(filenames), self.batch_size):
            end = start + self.batch_size
            batch_files = filenames[start:end]

            imgs = []
            for fname in batch_files:
                try:
                    img_path = os.path.join(self.input_folder, fname)
                    img = Image.open(img_path).convert("RGB")
                    imgs.append(np.asarray(img))
                except Exception as e:
                    logging.error(f"Error loading {fname}: {e}")
                    continue

            if not imgs:
                continue

            arr = np.stack(imgs, axis=0)

            batch50 = self.darken_batch(arr, 50)
            batch80 = self.darken_batch(arr, 80)

            for i, fname in enumerate(batch_files[: len(imgs)]):
                name, ext = os.path.splitext(fname)

                out50_path = os.path.join(
                    self.preprocessed_folder, f"{name}_50_darkened{ext}"
                )
                Image.fromarray(batch50[i]).save(out50_path, "JPEG", quality=95)

                out80_path = os.path.join(
                    self.preprocessed_folder, f"{name}_80_darkened{ext}"
                )
                Image.fromarray(batch80[i]).save(out80_path, "JPEG", quality=95)

            batch_num = start // self.batch_size + 1
            total_batches = (len(filenames) + self.batch_size - 1) // self.batch_size
            logging.info(f"Processed batch {batch_num}/{total_batches}")

        elapsed = time.time() - start_time
        logging.info(
            f"✓ Darkening complete! Processed {len(filenames)} images in {elapsed:.2f}s"
        )
        logging.info(f"  Created {len(filenames) * 2} darkened images")

    def enhance_classical(self, img, gamma=1.5, denoise_strength=5):
        """Apply classical enhancement (CLAHE + gamma + denoising)."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        gamma_inv = 1.0 / gamma
        table = np.array([(i / 255.0) ** gamma_inv * 255 for i in range(256)]).astype(
            "uint8"
        )
        enhanced_gamma = cv2.LUT(enhanced, table)

        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced_gamma,
            None,
            h=denoise_strength,
            hColor=denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        return denoised

    def enhance_classical_batch(self):
        """Stage 2A: Apply classical enhancement to darkened images."""
        if not self.enable_classical:
            logging.info("Classical enhancement stage skipped")
            return

        logging.info("=" * 60)
        logging.info("STAGE 2A: CLASSICAL ENHANCEMENT")
        logging.info("=" * 60)

        start_time = time.time()

        filenames = [
            f
            for f in os.listdir(self.preprocessed_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "_darkened" in f
        ]

        if not filenames:
            logging.warning(f"No darkened images found in {self.preprocessed_folder}")
            return

        logging.info(f"Found {len(filenames)} darkened images to enhance")

        processed = 0
        for start in range(0, len(filenames), self.batch_size):
            end = start + self.batch_size
            batch_files = filenames[start:end]

            for fname in batch_files:
                try:
                    img_path = os.path.join(self.preprocessed_folder, fname)
                    img = cv2.imread(img_path)
                    if img is None:
                        logging.error(f"Error loading: {img_path}")
                        continue

                    enhanced_img = self.enhance_classical(img)

                    name, ext = os.path.splitext(fname)
                    out_path = os.path.join(
                        self.preprocessed_folder, f"{name}_classical{ext}"
                    )
                    cv2.imwrite(out_path, enhanced_img)
                    processed += 1
                except Exception as e:
                    logging.error(f"Error processing {fname}: {e}")

            batch_num = start // self.batch_size + 1
            total_batches = (len(filenames) + self.batch_size - 1) // self.batch_size
            logging.info(f"Processed batch {batch_num}/{total_batches}")

        elapsed = time.time() - start_time
        logging.info(
            f"✓ Classical enhancement complete! Processed {processed} images in {elapsed:.2f}s"
        )

    def enhance_deep_batch(self, img_list):
        """Enhance a batch using Zero-DCE."""
        imgs_norm = np.array([img.astype(np.float32) / 255.0 for img in img_list])
        img_input = torch.from_numpy(np.transpose(imgs_norm, (0, 3, 1, 2))).to(
            self.device
        )

        with torch.no_grad():
            _, enhanced_batch, _ = self.model(img_input)

        enhanced_imgs = enhanced_batch.cpu().numpy()
        enhanced_imgs = np.clip(
            np.transpose(enhanced_imgs, (0, 2, 3, 1)) * 255, 0, 255
        ).astype(np.uint8)

        return enhanced_imgs

    def enhance_deep_learning(self):
        """Stage 2B: Apply deep learning enhancement to darkened images."""
        if not self.enable_deep or self.model is None:
            logging.info("Deep learning enhancement stage skipped")
            return

        logging.info("=" * 60)
        logging.info("STAGE 2B: DEEP LEARNING ENHANCEMENT (Zero-DCE)")
        logging.info("=" * 60)

        start_time = time.time()

        filenames = [
            f
            for f in os.listdir(self.preprocessed_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and "_darkened" in f
            and "_classical" not in f
        ]

        if not filenames:
            logging.warning(f"No darkened images found in {self.preprocessed_folder}")
            return

        logging.info(f"Found {len(filenames)} darkened images to enhance")

        processed = 0
        for start in range(0, len(filenames), self.batch_size):
            end = start + self.batch_size
            batch_files = filenames[start:end]

            imgs = []
            valid_fnames = []
            for fname in batch_files:
                try:
                    img_path = os.path.join(self.preprocessed_folder, fname)
                    img = cv2.imread(img_path)
                    if img is not None:
                        imgs.append(img)
                        valid_fnames.append(fname)
                except Exception as e:
                    logging.error(f"Error loading {fname}: {e}")

            if not imgs:
                continue

            try:
                enhanced_imgs = self.enhance_deep_batch(imgs)

                for fname, enhanced_img in zip(valid_fnames, enhanced_imgs):
                    name, ext = os.path.splitext(fname)
                    out_path = os.path.join(
                        self.preprocessed_folder, f"{name}_deep{ext}"
                    )
                    cv2.imwrite(out_path, enhanced_img)
                    processed += 1
            except Exception as e:
                logging.error(f"Error in deep enhancement batch: {e}")

            batch_num = start // self.batch_size + 1
            total_batches = (len(filenames) + self.batch_size - 1) // self.batch_size
            logging.info(f"Processed batch {batch_num}/{total_batches}")

        elapsed = time.time() - start_time
        logging.info(
            f"✓ Deep learning enhancement complete! Processed {processed} images in {elapsed:.2f}s"
        )

    def draw_facial_landmarks(self, image_path, output_path=None):
        """Detect faces and draw facial landmarks."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)

        if not detections:
            return image, 0

        for detection in detections:
            x, y, width, height = detection["box"]
            confidence = detection["confidence"]

            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Conf: {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            keypoints = detection["keypoints"]
            landmarks = {
                "left_eye": (keypoints["left_eye"], (255, 0, 0)),
                "right_eye": (keypoints["right_eye"], (0, 255, 255)),
                "nose": (keypoints["nose"], (0, 0, 255)),
                "mouth_left": (keypoints["mouth_left"], (255, 255, 0)),
                "mouth_right": (keypoints["mouth_right"], (255, 0, 255)),
            }

            for name, (point, color) in landmarks.items():
                cv2.circle(image, point, 5, color, -1)
                cv2.putText(
                    image,
                    name.replace("_", " ").title(),
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )

        if output_path:
            cv2.imwrite(output_path, image)

        return image, len(detections)

    def process_face_detection(self):
        """Stage 3: Detect faces in all preprocessed images."""
        if not self.enable_detection or self.detector is None:
            logging.info("Face detection stage skipped")
            return

        logging.info("=" * 60)
        logging.info("STAGE 3: FACE DETECTION AND LANDMARK ANNOTATION")
        logging.info("=" * 60)

        start_time = time.time()

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        input_path = Path(self.preprocessed_folder)
        image_files = [
            f
            for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            logging.warning(f"No images found in {self.preprocessed_folder}")
            return

        logging.info(f"Found {len(image_files)} images to process")

        total_faces = 0
        images_with_faces = 0
        processed = 0

        for i, image_file in enumerate(image_files, 1):
            try:
                output_path = os.path.join(
                    self.results_folder, f"detected_{image_file.name}"
                )
                result_image, num_faces = self.draw_facial_landmarks(
                    str(image_file), output_path
                )

                total_faces += num_faces
                if num_faces > 0:
                    images_with_faces += 1

                processed += 1

                if i % 10 == 0 or i == len(image_files):
                    logging.info(f"Processed {i}/{len(image_files)} images")

            except Exception as e:
                logging.error(f"Error processing {image_file.name}: {e}")

        elapsed = time.time() - start_time
        logging.info(f"✓ Face detection complete!")
        logging.info(f"  Processed {processed} images in {elapsed:.2f}s")
        logging.info(f"  Detected {total_faces} faces in {images_with_faces} images")
        logging.info(f"  Results saved to: {self.results_folder}")

    def run(self):
        """Execute the full pipeline."""
        pipeline_start = time.time()

        logging.info("Starting Image Processing Pipeline")
        logging.info(f"Input folder: {self.input_folder}")
        logging.info(f"Output base: {self.output_base}")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info("")

        try:
            if self.enable_darkening:
                self.darken_images()

            if self.enable_classical:
                self.enhance_classical_batch()

            if self.enable_deep:
                self.enhance_deep_learning()

            if self.enable_detection:
                self.process_face_detection()

            total_time = time.time() - pipeline_start
            logging.info("=" * 60)
            logging.info("PIPELINE COMPLETE!")
            logging.info(
                f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)"
            )
            logging.info("=" * 60)

        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Image Processing Pipeline for Face Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with all stages
  python image_processing_pipeline.py --input ./images --output ./output
  
  # Only darkening and detection (skip enhancement)
  python image_processing_pipeline.py --input ./images --no-classical --no-deep
  
  # Custom batch size for faster processing
  python image_processing_pipeline.py --input ./images --batch-size 64
        """,
    )

    parser.add_argument(
        "--input", "-i", required=True, help="Input folder containing images to process"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output base directory (default: output)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--no-darkening", action="store_true", help="Skip darkening stage"
    )
    parser.add_argument(
        "--no-classical", action="store_true", help="Skip classical enhancement"
    )
    parser.add_argument(
        "--no-deep", action="store_true", help="Skip deep learning enhancement"
    )
    parser.add_argument(
        "--no-detection", action="store_true", help="Skip face detection"
    )
    parser.add_argument(
        "--max-display",
        type=int,
        default=5,
        help="Maximum images to save visualizations for (default: 5)",
    )

    args = parser.parse_args()

    # Validate input folder
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        sys.exit(1)

    # Create and run pipeline
    pipeline = ImageProcessingPipeline(
        input_folder=args.input,
        output_base=args.output,
        batch_size=args.batch_size,
        enable_darkening=not args.no_darkening,
        enable_classical=not args.no_classical,
        enable_deep=not args.no_deep,
        enable_detection=not args.no_detection,
        max_display=args.max_display,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
