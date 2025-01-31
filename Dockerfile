# Use the official Ultralytics image with GPU support
FROM ultralytics/ultralytics:latest

# Set the working directory
WORKDIR /app


# Define build arguments for Kaggle credentials
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

# Create the .kaggle directory and generate the kaggle.json file
RUN mkdir -p /root/.kaggle && \
    echo '{"username":"'$KAGGLE_USERNAME'","key":"'$KAGGLE_KEY'"}' > /root/.kaggle/kaggle.json && \
    chmod 600 /root/.kaggle/kaggle.json

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep the container running
CMD ["tail", "-f", "/dev/null"]