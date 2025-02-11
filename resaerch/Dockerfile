FROM pytorch/pytorch

# Set the working directory in the container
WORKDIR /app

RUN apt-get upgrade && apt-get update -y

# Copy the requirements file into the container
COPY requirements.txt .
# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt
