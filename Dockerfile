FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model
COPY app.py .
COPY model.joblib .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
