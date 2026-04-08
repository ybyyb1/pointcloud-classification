#!/usr/bin/env python3
"""
Deployment script for Point Cloud Classification System.
Prepares models for production deployment, creates configuration files,
and sets up the deployment environment.
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from config import SystemConfig, save_config


class DeploymentManager:
    """Manages deployment of point cloud classification models."""

    def __init__(self, output_dir: str = "./deployment"):
        """
        Initialize deployment manager.

        Args:
            output_dir: Directory where deployment artifacts will be saved
        """
        self.output_dir = Path(output_dir)
        self.artifacts_dir = self.output_dir / "artifacts"
        self.configs_dir = self.output_dir / "configs"
        self.models_dir = self.output_dir / "models"
        self.examples_dir = self.output_dir / "examples"

        # Create directories
        self._create_directories()

    def _create_directories(self):
        """Create deployment directory structure."""
        directories = [
            self.output_dir,
            self.artifacts_dir,
            self.configs_dir,
            self.models_dir,
            self.examples_dir,
            self.examples_dir / "data",
            self.examples_dir / "scripts"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def create_configuration_templates(self):
        """Create configuration file templates for different deployment scenarios."""
        print("Creating configuration templates...")

        # 1. Production configuration
        prod_config = SystemConfig()
        prod_config.project_name = "Point Cloud Classification - Production"
        prod_config.version = "1.0.0"

        # Optimize for production
        prod_config.training.batch_size = 64
        prod_config.training.use_amp = True  # Mixed precision
        prod_config.training.num_workers = 4
        prod_config.training.pin_memory = True

        save_config(prod_config, self.configs_dir / "production_config.yaml")

        # 2. Development configuration
        dev_config = SystemConfig()
        dev_config.project_name = "Point Cloud Classification - Development"
        dev_config.training.batch_size = 16
        dev_config.training.epochs = 50
        dev_config.training.use_amp = False
        dev_config.training.num_workers = 2

        save_config(dev_config, self.configs_dir / "development_config.yaml")

        # 3. Inference-only configuration
        from config import InferenceConfig
        infer_config = InferenceConfig()
        infer_config.batch_size = 1  # Real-time inference
        infer_config.use_gpu = True
        infer_config.model_path = "./models/best_model.pth"

        infer_config_dict = infer_config.to_dict()
        with open(self.configs_dir / "inference_config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(infer_config_dict, f, default_flow_style=False, allow_unicode=True)

        # 4. Docker configuration
        docker_config = {
            "image": "pointcloud-classification:latest",
            "ports": ["8501:8501"],  # Streamlit port
            "volumes": [
                "./data:/app/data",
                "./models:/app/models",
                "./logs:/app/logs"
            ],
            "environment": {
                "PYTHONPATH": "/app",
                "CUDA_VISIBLE_DEVICES": "0",
                "MODEL_PATH": "/app/models/best_model.pth",
                "LOG_LEVEL": "INFO"
            },
            "command": "streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
        }

        with open(self.configs_dir / "docker_config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(docker_config, f, default_flow_style=False, allow_unicode=True)

        print(f"  Configuration templates saved to: {self.configs_dir}")

    def create_deployment_scripts(self):
        """Create deployment scripts for different environments."""
        print("Creating deployment scripts...")

        # 1. Docker deployment script
        dockerfile_content = """# Point Cloud Classification System Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/checkpoints

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

        with open(self.artifacts_dir / "Dockerfile", 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)

        # 2. Docker compose file
        docker_compose_content = """# Docker Compose for Point Cloud Classification
version: '3.8'

services:
  pointcloud-classification:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: pointcloud-classification
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
    environment:
      - PYTHONPATH=/app
      - CUDA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/models/best_model.pth
      - LOG_LEVEL=INFO
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Optional: Add Redis for caching
  redis:
    image: redis:alpine
    container_name: pointcloud-classification-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
"""

        with open(self.artifacts_dir / "docker-compose.yml", 'w', encoding='utf-8') as f:
            f.write(docker_compose_content)

        # 3. Deployment script for cloud providers
        deploy_sh_content = """#!/bin/bash
# Deployment script for Point Cloud Classification System

set -e

# Configuration
DEPLOYMENT_ENV=${1:-"production"}
MODEL_NAME=${2:-"point_transformer"}
GPU_TYPE=${3:-"T4"}

echo "Deploying Point Cloud Classification System..."
echo "Environment: $DEPLOYMENT_ENV"
echo "Model: $MODEL_NAME"
echo "GPU: $GPU_TYPE"

# Create deployment directory
DEPLOY_DIR="./deployments/$DEPLOYMENT_ENV"
mkdir -p $DEPLOY_DIR

# Copy necessary files
cp -r configs $DEPLOY_DIR/
cp -r scripts $DEPLOY_DIR/
cp main.py $DEPLOY_DIR/
cp requirements.txt $DEPLOY_DIR/
cp README.md $DEPLOY_DIR/

# Create model directory
mkdir -p $DEPLOY_DIR/models

# Download or copy model checkpoint
if [ -f "./models/$MODEL_NAME.pth" ]; then
    cp "./models/$MODEL_NAME.pth" $DEPLOY_DIR/models/
else
    echo "Warning: Model checkpoint not found at ./models/$MODEL_NAME.pth"
    echo "You will need to train or download the model separately."
fi

# Create configuration
CONFIG_FILE="$DEPLOY_DIR/configs/deployment_config.yaml"
cat > $CONFIG_FILE << EOF
# Deployment Configuration
# Generated for $DEPLOYMENT_ENV environment

project_name: "Point Cloud Classification - $DEPLOYMENT_ENV"
version: "1.0.0"

deployment:
  environment: "$DEPLOYMENT_ENV"
  model: "$MODEL_NAME"
  gpu_type: "$GPU_TYPE"
  timestamp: "$(date)"

inference:
  batch_size: 1
  use_gpu: true
  model_path: "./models/$MODEL_NAME.pth"

logging:
  level: "INFO"
  file: "./logs/inference.log"
EOF

echo "Deployment package created at: $DEPLOY_DIR"
echo "To run with Docker:"
echo "  cd $DEPLOY_DIR && docker build -t pointcloud-classification ."
echo "  docker run -p 8501:8501 pointcloud-classification"
"""

        with open(self.artifacts_dir / "deploy.sh", 'w', encoding='utf-8') as f:
            f.write(deploy_sh_content)

        # Make script executable
        os.chmod(self.artifacts_dir / "deploy.sh", 0o755)

        # 4. Kubernetes deployment file
        k8s_deployment_content = """# Kubernetes Deployment for Point Cloud Classification
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pointcloud-classification
  labels:
    app: pointcloud-classification
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pointcloud-classification
  template:
    metadata:
      labels:
        app: pointcloud-classification
    spec:
      containers:
      - name: pointcloud-classification
        image: pointcloud-classification:latest
        ports:
        - containerPort: 8501
        env:
        - name: PYTHONPATH
          value: "/app"
        - name: MODEL_PATH
          value: "/app/models/best_model.pth"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc
---
# Service
apiVersion: v1
kind: Service
metadata:
  name: pointcloud-classification-service
spec:
  selector:
    app: pointcloud-classification
  ports:
  - protocol: TCP
    port: 8501
    targetPort: 8501
  type: LoadBalancer
---
# Persistent Volume Claims
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
"""

        with open(self.artifacts_dir / "kubernetes-deployment.yaml", 'w', encoding='utf-8') as f:
            f.write(k8s_deployment_content)

        print(f"  Deployment scripts saved to: {self.artifacts_dir}")

    def create_example_data_and_scripts(self):
        """Create example data and scripts for testing deployment."""
        print("Creating example data and scripts...")

        # 1. Example inference script
        inference_script = """#!/usr/bin/env python3
"""
        with open(self.examples_dir / "scripts" / "run_inference.py", 'w', encoding='utf-8') as f:
            f.write(inference_script)

        # 2. Example batch inference script
        batch_inference_script = """#!/usr/bin/env python3
"""
        with open(self.examples_dir / "scripts" / "batch_inference.py", 'w', encoding='utf-8') as f:
            f.write(batch_inference_script)

        # 3. Example API server (Flask)
        api_server_script = """#!/usr/bin/env python3
"""
        with open(self.examples_dir / "scripts" / "api_server.py", 'w', encoding='utf-8') as f:
            f.write(api_server_script)

        print(f"  Example scripts saved to: {self.examples_dir / 'scripts'}")

    def package_deployment(self, package_name: str = "pointcloud-classification-deployment"):
        """Package deployment artifacts into a tar.gz file."""
        print(f"Packaging deployment artifacts as {package_name}.tar.gz...")

        package_path = self.output_dir / f"{package_name}.tar.gz"

        with tarfile.open(package_path, "w:gz") as tar:
            # Add artifacts directory
            tar.add(self.artifacts_dir, arcname="artifacts")

            # Add configs directory
            tar.add(self.configs_dir, arcname="configs")

            # Add examples directory
            tar.add(self.examples_dir, arcname="examples")

            # Add README
            readme_path = self.output_dir / "DEPLOYMENT_README.md"
            if readme_path.exists():
                tar.add(readme_path, arcname="DEPLOYMENT_README.md")

        print(f"  Deployment package created: {package_path}")
        print(f"  Size: {package_path.stat().st_size / 1024 / 1024:.2f} MB")

    def create_readme(self):
        """Create deployment README file."""
        readme_content = """# Point Cloud Classification System - Deployment Guide

This directory contains deployment artifacts for the Point Cloud Classification System.

## Directory Structure

```
deployment/
├── artifacts/          # Deployment scripts and files
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── deploy.sh
│   └── kubernetes-deployment.yaml
├── configs/           # Configuration templates
│   ├── production_config.yaml
│   ├── development_config.yaml
│   ├── inference_config.yaml
│   └── docker_config.yaml
├── examples/          # Example data and scripts
│   ├── data/
│   └── scripts/
└── DEPLOYMENT_README.md
```

## Deployment Options

### 1. Docker Deployment (Recommended)

```bash
# Build the Docker image
docker build -t pointcloud-classification -f artifacts/Dockerfile .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  pointcloud-classification
```

### 2. Docker Compose

```bash
# Start all services
docker-compose -f artifacts/docker-compose.yml up -d

# View logs
docker-compose -f artifacts/docker-compose.yml logs -f
```

### 3. Kubernetes

```bash
# Apply Kubernetes configuration
kubectl apply -f artifacts/kubernetes-deployment.yaml

# Check deployment status
kubectl get pods
kubectl get services
```

### 4. Manual Deployment

```bash
# Run the deployment script
./artifacts/deploy.sh production point_transformer T4

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Configuration

Edit the configuration files in `configs/` for your environment:

- `production_config.yaml`: Production settings
- `development_config.yaml`: Development settings
- `inference_config.yaml`: Inference-only settings
- `docker_config.yaml`: Docker-specific settings

## Model Deployment

1. Place your trained model checkpoints in the `models/` directory
2. Update the model path in configuration files
3. Ensure the model matches the expected input format (1024 points, 3 channels)

## Monitoring and Logging

- Application logs are written to `/app/logs` in the container
- Streamlit logs are available in the container stdout
- GPU utilization can be monitored with `nvidia-smi`

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in configuration
2. **Model Loading Failed**: Check model path and format
3. **Port Already in Use**: Change the port in Dockerfile or command
4. **Missing Dependencies**: Run `pip install -r requirements.txt`

### Getting Help

- Check the application logs
- Review configuration files
- Consult the main project README
- Open an issue on GitHub

## Next Steps

1. Test with sample data using scripts in `examples/scripts/`
2. Monitor performance and adjust configurations
3. Set up CI/CD pipeline for automated deployment
4. Configure monitoring and alerting

---
*Deployment package generated on $(date)*
"""

        readme_path = self.output_dir / "DEPLOYMENT_README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"  README created: {readme_path}")

    def run(self, package: bool = False):
        """Run the complete deployment preparation process."""
        print("=" * 60)
        print("Preparing Deployment Artifacts")
        print("=" * 60)

        self.create_configuration_templates()
        self.create_deployment_scripts()
        self.create_example_data_and_scripts()
        self.create_readme()

        if package:
            self.package_deployment()

        print("\n" + "=" * 60)
        print("Deployment Preparation Complete!")
        print("=" * 60)
        print(f"\nDeployment artifacts are ready in: {self.output_dir}")
        print("\nNext steps:")
        print("1. Review configuration files in configs/")
        print("2. Update model paths and settings as needed")
        print("3. Run: docker build -f artifacts/Dockerfile -t pointcloud-classification .")
        print("4. Deploy using your preferred method (Docker, Kubernetes, etc.)")
        print("\nFor detailed instructions, see: DEPLOYMENT_README.md")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare deployment artifacts for Point Cloud Classification System"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./deployment",
        help="Output directory for deployment artifacts (default: ./deployment)"
    )

    parser.add_argument(
        "--package",
        action="store_true",
        help="Package deployment artifacts into a tar.gz file"
    )

    parser.add_argument(
        "--package-name",
        type=str,
        default="pointcloud-classification-deployment",
        help="Name for the deployment package (default: pointcloud-classification-deployment)"
    )

    args = parser.parse_args()

    # Create deployment manager
    manager = DeploymentManager(output_dir=args.output_dir)

    # Run deployment preparation
    manager.run(package=args.package)


if __name__ == "__main__":
    main()