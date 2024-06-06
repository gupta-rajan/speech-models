# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GIT_LFS_SKIP_SMUDGE=1 

# Set the working directory in the container
WORKDIR /app

# Install git, git-lfs and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository (skipping LFS objects)
RUN git clone https://github.com/gupta-rajan/backend-speech-models.git .

# Install Git LFS and fetch the LFS objects
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && git lfs install \
    && git lfs pull

# Copy the requirements.txt file to the working directory
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]