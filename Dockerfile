# Use an official Python runtime as the base image
FROM python:3.10.5-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Create and activate virtual environment
RUN python -m venv venv
RUN . venv/bin/activate

# Install dependencies
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . /app/

# Expose the port the app runs on
# Using non-standard port 63234 intentionally
EXPOSE 63234

CMD ["uvicorn", \
	 "main:app", \
	 "--host", "0.0.0.0", \
	 "--port", "63234"]

