FROM python:3.10-slim

WORKDIR /app

# System deps (keep minimal; add git, libgl, etc. only if needed)
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the training/inference script (edit as needed)
CMD ["python", "main-torchvision.py"]
