# Tanglaw - Image Processing & Face Recognition API

A Flask REST API for image processing, face detection, and face recognition using RetinaFace->ArcFace and MTCNN->MobileFaceNet.

## Features

- **Image Processing**

  - Image Darkening: Simulates low-light conditions (50% and 80% levels)
  - Classical Enhancement: CLAHE + Gamma correction + Denoising
  - Deep Learning Enhancement: Zero-DCE neural network for low-light enhancement

- **Face Detection**

  - MTCNN: Fast, lightweight detector for MobileFaceNet pipeline
  - RetinaFace: High-accuracy detector integrated with ArcFace
  - 5-Point Landmarks: Eye, nose, and mouth corner detection
  - Bounding Boxes: Precise face localization with confidence scores

- **Face Recognition**
  - Two Models Available:
    - ArcFace (InsightFace): State-of-the-art accuracy, production-ready
    - MobileFaceNet: Lightweight, optimized for mobile/edge devices
  - Face Enrollment: Store face embeddings with names in either model
  - Face Recognition: Identify enrolled faces using selected model
  - Model Switching: Switch between models on-the-fly
  - Model Comparison: Compare results from both models on the same image

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python server.py
```

Server runs on `http://localhost:5000` by default.

### 3. Test the API

```bash
# Health check (shows available models)
curl http://localhost:5000/recognition/health

# Get available models
curl http://localhost:5000/recognition/models

# Switch to MobileFaceNet
curl -X POST http://localhost:5000/recognition/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "mobilefacenet"}'

# Enroll a face (uses current model)
curl -X POST http://localhost:5000/recognition/enroll \
  -F "image=@photo.jpg" \
  -F "name=John Doe"

# Enroll with specific model
curl -X POST http://localhost:5000/recognition/enroll \
  -F "image=@photo.jpg" \
  -F "name=Jane Doe" \
  -F "model=arcface"

# Recognize faces
curl -X POST http://localhost:5000/recognition/recognize \
  -F "image=@group_photo.jpg"

# Compare both models
curl -X POST http://localhost:5000/recognition/compare \
  -F "image=@test_photo.jpg"
```

## API Endpoints

### Image Processing

- `POST /process` - Process image through full pipeline

### Face Recognition

- `GET /recognition/health` - Service health check (shows available models)
- `GET /recognition/models` - Get available models and current model
- `POST /recognition/models/switch` - Switch active model
- `POST /recognition/enroll` - Enroll new face (optional model parameter)
- `POST /recognition/recognize` - Recognize faces in image (optional model parameter)
- `POST /recognition/compare` - Compare results from both models
- `GET /recognition/list` - List all enrolled faces
- `DELETE /recognition/remove/<name>` - Remove specific face
- `POST /recognition/clear` - Clear all faces

## Configuration

### Server Options

```bash
python server.py --host 0.0.0.0 --port 8000 --debug
```

### Environment Variables

Configuration in `config/settings.py`:

- `UPLOAD_FOLDER`: Temporary upload directory
- `MAX_CONTENT_LENGTH`: Maximum file size (16MB default)
- `ALLOWED_EXTENSIONS`: Accepted image formats

## Project Structure

```
server/
├── config/              # Configuration settings
├── models/              # Data models and persistence
├── services/            # Business logic services
│   ├── arcface_recognizer.py           # ArcFace recognition engine
│   ├── mobilefacenet_recognizer.py     # MobileFaceNet recognition engine
│   ├── mtcnn_detector.py               # MTCNN face detection
│   ├── face_recognition_service.py     # Service layer
│   └── image_processor.py              # Image enhancement
├── routes/              # API endpoints
├── utils/               # Utility functions
├── database/            # Face embeddings storage
├── Zero-DCE/            # Deep learning enhancement model
└── server.py            # Application entry point
```

## Dependencies

- **Flask**: Python framework for API
- **InsightFace**: Face recognition models (ArcFace, MobileFaceNet, RetinaFace)
- **OpenCV**: Image processing
- **MTCNN**: Lightweight face detection
- **PyTorch**: Zero-DCE deep learning enhancement
- **NumPy**: Numerical operations

## Model Information

### Face Detection Models

#### RetinaFace (InsightFace)

- **Accuracy**: High precision face detection
- **Speed**: Moderate
- **Features**: 5-point landmarks, bounding boxes
- **Pipeline**: Used with ArcFace recognition

#### MTCNN

- **Accuracy**: Good for most scenarios
- **Speed**: Fast, lightweight
- **Features**: 5-point landmarks, bounding boxes
- **Pipeline**: Used with MobileFaceNet recognition

### Face Recognition Models

#### ArcFace (buffalo_l)

- **Accuracy**: State-of-the-art
- **Embedding Size**: 512 dimensions
- **Default Threshold**: 0.4
- **Face Detector**: RetinaFace (built-in)
- **Use Case**: Production environments requiring high accuracy

#### MobileFaceNet (antelopev2)

- **Accuracy**: Good (lightweight)
- **Embedding Size**: 128 dimensions
- **Default Threshold**: 0.70
- **Face Detector**: MTCNN
- **Use Case**: Mobile/edge devices, real-time applications

## Logging

Server logs saved to `server.log` with:

- Request tracking with unique IDs
- Error messages and stack traces
- Model initialization status
- Recognition results and timing

## License

MIT License - See LICENSE file for details
