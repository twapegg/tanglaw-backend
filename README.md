# Tanglaw Server - Image Processing & Face Recognition API

A production-ready Flask REST API for image processing, face detection, and face recognition with dual-model support: InsightFace ArcFace and MobileFaceNet.

## Features

### Image Processing

- **Image Darkening**: Simulates low-light conditions (50% and 80% levels)
- **Classical Enhancement**: CLAHE + Gamma correction + Denoising
- **Deep Learning Enhancement**: Zero-DCE neural network for low-light enhancement
- **Face Detection**: MTCNN-based detection with 5-point landmarks

### Face Recognition (Dual Model Support)

- **Two Models Available**:
  - **ArcFace (InsightFace)**: State-of-the-art accuracy, production-ready
  - **MobileFaceNet**: Lightweight, optimized for mobile/edge devices
- **Face Enrollment**: Store face embeddings with names in either model
- **Face Recognition**: Identify enrolled faces using selected model
- **Model Switching**: Switch between models on-the-fly
- **Model Comparison**: Compare results from both models on the same image
- **Database Management**: Separate databases for each model

### API Features

- **RESTful API**: Clean, documented endpoints
- **Batch Processing**: Efficient handling of multiple images
- **Comprehensive Logging**: Request tracking and error logging
- **Graceful Degradation**: Both models optional if dependencies unavailable
- **Flexible Model Selection**: Use different models per request or switch globally

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

### 4. Run Test Suite

```bash
python test_recognition.py
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

See [FACE_RECOGNITION_GUIDE.md](FACE_RECOGNITION_GUIDE.md) for detailed dual-model documentation.

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
│   ├── face_detector.py                # MTCNN face detection
│   ├── face_recognition_service.py     # Service layer
│   └── image_processor.py              # Image enhancement
├── routes/              # API endpoints
├── utils/               # Utility functions
├── cleanup.py           # Maintenance script
└── server.py            # Application entry point
```

## Dependencies

- **Flask**: Web framework
- **InsightFace**: Face recognition models (ArcFace, MobileFaceNet)
- **OpenCV**: Image processing
- **MTCNN**: Face detection
- **PyTorch**: Zero-DCE deep learning enhancement
- **NumPy**: Numerical operations

## Model Information

### ArcFace (buffalo_l)

- **Accuracy**: State-of-the-art
- **Embedding Size**: 512 dimensions
- **Default Threshold**: 0.4
- **Use Case**: Production environments requiring high accuracy

### MobileFaceNet (antelopev2)

- **Accuracy**: Good (lightweight)
- **Embedding Size**: 128 dimensions
- **Default Threshold**: 0.70
- **Use Case**: Mobile/edge devices, real-time applications

## Troubleshooting

### Face Recognition Issues

**No faces detected:**

- Ensure image has clear, frontal faces
- Check image quality and lighting
- Minimum face size requirements apply

**Low confidence scores:**

- Try adjusting the threshold parameter
- Re-enroll with multiple images of the same person
- Ensure consistent lighting conditions

**Model not available:**

- Check InsightFace installation: `pip install insightface`
- Verify model downloads completed (first run downloads models)

### Server Issues

**Port already in use:**

```bash
python server.py --port 8000
```

**Out of memory:**

- Reduce batch processing size
- Use MobileFaceNet instead of ArcFace
- Close unnecessary applications

## Logging

Server logs saved to `server.log` with:

- Request tracking with unique IDs
- Error messages and stack traces
- Model initialization status
- Recognition results and timing

## License

MIT License - See LICENSE file for details
