FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir huggingface_hub

# Copy app code
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints data/uploads reports data

# Download model weights from Hugging Face at build time
RUN python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='Newt-otter/abyssia-edna', filename='ft_final.pt')
shutil.copy(path, 'checkpoints/ft_final.pt')
path2 = hf_hub_download(repo_id='Newt-otter/abyssia-edna', filename='label_to_id.json')
shutil.copy(path2, 'checkpoints/label_to_id.json')
"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
