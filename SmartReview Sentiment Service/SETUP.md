# Setup Instructions

Complete step-by-step guide to set up and run the SmartReview Sentiment Service.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **pip** package manager
- **Node.js 16+** and **npm** (for frontend development)
- **Git** (for version control)
- **Docker & Docker Compose** (optional, for containerized deployment)

### Verify Installations

```bash
python --version    # Should be 3.8 or higher
pip --version
node --version      # Should be 16 or higher
npm --version
git --version
```

## 🚀 Step-by-Step Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "SmartReview Sentiment Service"
```

### Step 2: Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Prepare the Dataset

The dataset preparation script downloads the IMDB reviews dataset and processes it:

```bash
python src/data/prepare_dataset.py
```

**What this does:**
- Downloads IMDB reviews dataset (~80MB)
- Cleans and processes the text
- Splits into train (70%), validation (10%), and test (20%) sets
- Saves processed data as CSV files in `data/processed/`

**Expected output:**
```
Preparing dataset...
Loading reviews...
Loaded 25000 reviews

Dataset prepared successfully!
Train: 17500 samples (8750 positive, 8750 negative)
Validation: 2500 samples (1250 positive, 1250 negative)
Test: 5000 samples (2500 positive, 2500 negative)
```

### Step 4: Train the Model

Train the sentiment classification model:

```bash
python src/training/train.py
```

**For M2 Mac (recommended settings):**
```bash
python src/training/train.py --num-epochs 3 --batch-size 16
```

**Full training options:**
```bash
python src/training/train.py \
    --train-data data/processed/train.csv \
    --val-data data/processed/val.csv \
    --model-dir models \
    --num-epochs 5 \
    --batch-size 32 \
    --learning-rate 0.001
```

**What this does:**
- Builds vocabulary from training data
- Initializes LSTM model
- Trains for specified epochs
- Validates after each epoch
- Saves best model to `models/best_model.pt`
- Logs metrics to `models/training_metrics.csv`

**Expected training time:**
- M2 Mac: ~10-15 minutes for 3 epochs
- CPU: ~20-30 minutes for 3 epochs
- GPU: ~2-5 minutes for 3 epochs

### Step 5: Build the Frontend (Optional)

If you want to develop or modify the frontend:

```bash
cd frontend
npm install
npm run build
cd ..
```

The build output will be in `src/api/static/` for FastAPI to serve.

**For frontend development:**
```bash
cd frontend
npm run dev
# Opens on http://localhost:3000 with hot reload
```

### Step 6: Run the API Server

#### Option A: Using Python (Development)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload on code changes.

#### Option B: Using Docker Compose (Production)

```bash
docker-compose up
```

This builds and runs the containerized API.

#### Option C: Using Docker directly

```bash
docker build -t smartreview-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models smartreview-api
```

### Step 7: Verify Installation

1. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"healthy","message":"Model loaded and ready"}`

2. **Test prediction:**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "I love this product!"}'
   ```

3. **Open in browser:**
   - Frontend: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 🐳 Docker Setup

### Build and Run with Docker Compose

```bash
# Build and start
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t smartreview-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name smartreview-api \
  smartreview-api

# View logs
docker logs -f smartreview-api

# Stop container
docker stop smartreview-api
```

## 🔧 Configuration

### Environment Variables

Set these before running:

```bash
export MODEL_DIR=models          # Model directory path
export PORT=8000                 # API port
```

Or create a `.env` file:
```
MODEL_DIR=models
PORT=8000
```

### Model Directory Structure

Ensure your `models/` directory contains:
```
models/
├── best_model.pt    # Trained model weights
├── vocab.txt        # Vocabulary file
└── training_metrics.csv  # Training history (optional)
```

## 🧪 Testing

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Positive sentiment
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this service!"}'

# Negative sentiment
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is terrible"}'
```

### Test the Frontend

1. Open http://localhost:8000
2. Enter review text
3. Click "Analyze Sentiment"
4. View results with confidence scores

## 🐛 Troubleshooting

### Issue: Model not found

**Error:** `FileNotFoundError: Model not found at models/best_model.pt`

**Solution:** Train the model first:
```bash
python src/training/train.py
```

### Issue: Port already in use

**Error:** `Address already in use`

**Solution:** Use a different port:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

### Issue: Frontend not loading

**Error:** Blank page or 404 errors

**Solution:** Rebuild the frontend:
```bash
cd frontend
npm run build
cd ..
```

### Issue: Out of memory during training

**Error:** `RuntimeError: CUDA out of memory` or system freeze

**Solution:** Use smaller batch size:
```bash
python src/training/train.py --batch-size 8
```

### Issue: Dataset download fails

**Error:** Connection timeout or download error

**Solution:** 
- Check internet connection
- The dataset is ~80MB, ensure sufficient bandwidth
- Try running the script again

## 📊 Performance Optimization

### For Faster Training

1. **Use GPU** (if available):
   ```bash
   # PyTorch will automatically use CUDA if available
   python src/training/train.py
   ```

2. **Reduce batch size** for memory-constrained systems:
   ```bash
   python src/training/train.py --batch-size 16
   ```

3. **Train fewer epochs** for quick testing:
   ```bash
   python src/training/train.py --num-epochs 1
   ```

### For Production Deployment

1. **Use Docker** for consistent environments
2. **Enable GPU** if available for faster inference
3. **Use a reverse proxy** (nginx) for production
4. **Set up monitoring** and logging
5. **Use environment variables** for configuration

## 🔄 Updating the Model

To retrain with new data or different parameters:

```bash
# Prepare new dataset (if needed)
python src/data/prepare_dataset.py

# Train new model
python src/training/train.py --num-epochs 10

# Restart API to load new model
# (Stop and restart the server)
```

## 📚 Next Steps

After setup:

1. **Explore the API docs**: http://localhost:8000/docs
2. **Try different reviews** in the frontend
3. **Integrate the API** into your application
4. **Customize the model** for your use case
5. **Deploy to production** using Docker

## 💡 Tips

- **Development**: Use `--reload` flag for auto-reload
- **Production**: Use Docker for consistent deployment
- **Monitoring**: Check `/health` endpoint regularly
- **Logs**: Monitor training metrics in `models/training_metrics.csv`
- **Frontend**: Use `npm run dev` for hot reload during development

---

For more information, see the [README.md](README.md) file.

