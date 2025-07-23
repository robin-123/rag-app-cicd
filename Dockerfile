# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose a port if your application listens on one (e.g., for a web API)
# For this RAG app, it's a CLI, so we don't strictly need to expose a port
# EXPOSE 8000 

# Run app.py when the container launches
CMD ["python", "app.py"]
