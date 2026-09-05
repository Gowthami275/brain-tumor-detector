FROM python:3.10

WORKDIR /app

COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install \
    tensorflow==2.19.0 \
    keras==3.10.0 \
    numpy \
    gradio \
    opencv-python-headless \
    pillow \
    fpdf \
    matplotlib

# Expose port (Gradio uses 7860)
EXPOSE 7860

# Run app
CMD ["python", "app.py"]