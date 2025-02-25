from setuptools import setup, find_packages

setup(
    name="vyper-ai",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        # Core Dependencies
        "python-dotenv>=1.0.0",
        "rich>=13.5.0",
        "pydantic>=2.0.0",
        "PyYAML>=6.0.1",
        "aiohttp>=3.8.5",
        "asyncio>=3.4.3",
        "certifi>=2023.7.22",
        "typing-extensions>=4.7.1",
        
        # AI and Machine Learning
        "langchain>=0.0.300",
        "langchain-openai>=0.0.2",
        "openai>=1.3.0",
        "anthropic>=0.3.11",
        "transformers>=4.30.2",
        "torch>=2.0.1",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.0",
        
        # Document Processing
        "python-docx>=0.8.11",
        "python-pptx>=0.6.21",
        "reportlab>=4.0.4",
        "PyPDF2>=3.0.1",
        "pillow>=10.0.0",
        
        # Data Processing and Analysis
        "pandas>=2.0.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        
        # Web and API
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "httpx>=0.24.1",
        "requests>=2.31.0",
        "websockets>=11.0.3",
        
        # Security
        "cryptography>=41.0.3",
        "python-jose>=3.3.0",
        "passlib>=1.7.4",
        "bcrypt>=4.0.1",
        "python-multipart>=0.0.6",
        
        # Monitoring and Metrics
        "prometheus-client>=0.17.1",
        "grafana-api>=1.0.3",
        "psutil>=5.9.5",
        "statsd>=4.0.1",
        
        # Caching and Queues
        "redis>=5.0.0",
        "celery>=5.3.1",
        
        # Storage
        "boto3>=1.28.3",
        "azure-storage-blob>=12.17.0",
        "google-cloud-storage>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.1",
            "pytest-xdist>=3.3.1",
            "faker>=19.2.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.1",
            "isort>=5.12.0",
            "pre-commit>=3.3.3",
        ],
    },
)