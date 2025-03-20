# Use a Python base image
FROM python:3.9

# Install Java (OpenJDK) to enable 'javac' command
RUN apt-get update && apt-get install -y default-jdk

# Set working directory
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose Flask's port
EXPOSE 5000

# Run Flask application
CMD ["python", "app.py"]
