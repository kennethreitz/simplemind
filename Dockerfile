FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Create and set working directory
WORKDIR /app

# Copy requirements/project files
ONBUILD COPY . .

# Install dependencies using uv
RUN uv pip install "simplemind[full]" --system

# Set default command
CMD ["python"]
