# 🐕 Dog Breeds Classifier

A deep learning project to classify different dog breeds using state-of-the-art computer vision techniques.

## 📁 Project Structure
```
.
├── .venv/                  # Virtual environment directory
├── data/                   # Dataset storage
├── input/                  # Input data for predictions
├── logs/                   # Training and inference logs
├── models/                 # Saved model checkpoints
├── output/                 # Model predictions and evaluations
├── src/                    # Source code
├── .gitignore             # Git ignore file
├── .python-version        # Python version specification
├── docker-compose.yml     # Docker compose configuration
├── Dockerfile             # Docker configuration
├── pyproject.toml         # Project dependencies and metadata
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── uv.lock               # Dependency lock file
```

## 🎯 Features

- Multi-class classification of dog breeds
- Real-time prediction capabilities
- Support for transfer learning using state-of-the-art models
- Comprehensive data visualization and analysis tools
- Docker support for easy deployment

## 📊 Dataset Information

To use this classifier, you'll need to organize your dog breed dataset in the `data` directory. The dataset should be structured as follows:

```
data/
├── train/
│   ├── breed1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   ├── breed2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
├── val/
│   ├── breed1/
│   │   ├── image1.jpg
├── test/
    ├── breed1/
        ├── image1.jpg
```

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Git
- 8GB+ RAM recommended
- NVIDIA GPU (optional, but recommended for training)

### Running with Docker

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dog-breeds-classifier.git
cd dog-breeds-classifier
```

2. **Build and run using Docker Compose**
```bash
# Build the Docker image
docker-compose build

# Run the services
docker-compose up
```

3. **Access the application**
- Web interface: http://localhost:8080
- API endpoint: http://localhost:8080/api/predict

### Docker Configuration

The project includes two main services:

1. **Web Application**
```yaml
# Relevant section from docker-compose.yml
services:
  web:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
```

2. **Model Training (Optional)**
```yaml
  train:
    build: .
    command: ["python", "-m", "src.train"]
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 💻 Development Setup

1. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application locally**
```bash
python -m src.main
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Implementation inspired by various state-of-the-art papers
- Thanks to all contributors and the open-source community
