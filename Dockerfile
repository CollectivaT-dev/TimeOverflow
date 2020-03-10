# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Install postgres lib for the headers
RUN apt-get update && \
    apt-get -y install libpq-dev gcc

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app
RUN mkdir -p /app/results
VOLUME ./results /app/results

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt && \
    rm -r /root/.cache

# Run app.py when the container launches
CMD ["python", "testScript_grafos.py"]
