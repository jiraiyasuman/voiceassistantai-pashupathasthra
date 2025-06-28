FROM python:3.11-slim

# Install system dependencies for pyaudio, opencv, and other tools
RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Default run command
CMD ["python", "pashupathastra.py"]
