# SmartReview Sentiment Service

A production-ready REST API service that scores customer reviews (text) as positive/negative with confidence. Built with PyTorch + FastAPI, featuring a modern React frontend, and fully containerized for deployment.

## 🎯 Overview

Given any customer review text, return sentiment and confidence via REST API. The service uses a deep learning model (LSTM-based) trained on IMDB reviews dataset to provide accurate sentiment analysis.

**Live Demo**: Enter any review text and get instant sentiment classification with confidence scores.

## ✨ Features

- **🎨 Modern React Frontend**: Beautiful, responsive UI with Framer Motion animations
- **🤖 AI-Powered Analysis**: LSTM-based sentiment classifier with 84%+ accuracy
- **📊 Confidence Scoring**: Get detailed confidence scores for predictions
- **🚀 REST API**: Simple HTTP interface for easy integration
- **🐳 Dockerized**: Easy deployment with Docker and docker-compose
- **📈 Training Pipeline**: Complete data preparation and model training scripts
- **💎 Production Ready**: FastAPI-based with async support and health checks

## 🏗️ Project Structure

```
SmartReview Sentiment Service/
├── src/
│   ├── data/              # Data preparation scripts
│   │   └── prepare_dataset.py
│   ├── models/            # PyTorch model definitions
│   │   └── text_classifier.py
│   ├── training/          # Training scripts
│   │   └── train.py
│   └── api/               # FastAPI application
│       ├── main.py        # API server
│       └── static/        # React frontend (built)
├── frontend/              # React frontend source
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── styles/       # CSS files
│   │   └── utils/       # API utilities
│   └── package.json
├── models/                # Trained model weights
├── data/                  # Processed datasets
├── Dockerfile             # API container definition
├── docker-compose.yml     # Docker orchestration
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend development)
- Docker & Docker Compose (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "SmartReview Sentiment Service"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   ```bash
   python src/data/prepare_dataset.py
   ```
   This downloads the IMDB reviews dataset, processes it, and creates train/val/test splits.

4. **Train the model**
   ```bash
   python src/training/train.py --num-epochs 5 --batch-size 32
   ```
   For a quick test on M2 Mac, use smaller batches:
   ```bash
   python src/training/train.py --num-epochs 3 --batch-size 16
   ```

5. **Build the React frontend** (optional, for development)
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

6. **Run the API**
   ```bash
   # Using Python directly
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   
   # Or using Docker Compose
   docker-compose up
   ```

7. **Access the application**
   - Frontend: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## 📖 API Usage

### Endpoint: POST `/predict`

Analyze sentiment for a given text.

**Request:**
```json
{
  "text": "The delivery was slow but support was great"
}
```

**Response:**
```json
{
  "label": "positive",
  "score": 0.93
}
```

### Example with curl

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "The delivery was slow but support was great"}'
```

### Example with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I love this product! It exceeded my expectations."}
)

result = response.json()
print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2%}")
```

## 🎨 Frontend Development

The frontend is built with React 18 and Framer Motion for animations.

### Development Mode

```bash
cd frontend
npm install
npm run dev
```

This starts a development server on `http://localhost:3000` with hot reload.

### Building for Production

```bash
cd frontend
npm run build
```

The build output goes to `src/api/static/` for FastAPI to serve.

## 🐳 Docker Deployment

### Using Docker Compose

```bash
docker-compose up
```

This will:
- Build the API container
- Mount the models directory
- Expose the API on port 8000

### Using Docker directly

```bash
docker build -t smartreview-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models smartreview-api
```

## 📊 Model Training

### Training Configuration

The model uses:
- **Architecture**: Bidirectional LSTM with embedding layer
- **Embedding Dimension**: 128
- **Hidden Dimension**: 256
- **Layers**: 2
- **Dropout**: 0.3
- **Optimizer**: Adam
- **Learning Rate**: 0.001

### Training Metrics

Metrics are logged to:
- Console (stdout)
- CSV file: `models/training_metrics.csv`

The best model (based on validation F1 score) is saved to `models/best_model.pt`.

### Custom Training

```bash
python src/training/train.py \
    --train-data data/processed/train.csv \
    --val-data data/processed/val.csv \
    --model-dir models \
    --num-epochs 10 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --hidden-dim 256
```

## 🔧 Configuration

### Environment Variables

- `MODEL_DIR`: Directory containing model weights (default: `models`)
- `PORT`: API server port (default: `8000`)

### Model Files

The API expects these files in the model directory:
- `best_model.pt`: Trained model weights
- `vocab.txt`: Vocabulary file

## 📈 Performance

- **Validation Accuracy**: ~84.4%
- **Validation F1 Score**: ~0.844
- **Inference Speed**: <100ms per prediction (CPU)
- **Model Size**: ~47MB

## 🛠️ Development

### Project Commits

The project follows a structured commit history:

1. `chore: scaffold project structure` - Initial project setup
2. `feat: add dataset preparation script` - Data pipeline implementation
3. `feat: add PyTorch model and basic training pipeline` - Model and training
4. `feat: add FastAPI inference API and Docker setup` - API and deployment

### Adding Features

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Commit with descriptive messages
5. Push and create a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- IMDB Reviews Dataset from Stanford AI Lab
- PyTorch for deep learning framework
- FastAPI for the web framework
- React and Framer Motion for the frontend

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ using PyTorch, FastAPI, and React**
