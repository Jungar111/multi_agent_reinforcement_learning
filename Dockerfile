# Use Python 3.10 on Debian Bullseye as a parent image
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Set env vars
ENV POETRY_VERSION=1.5.1


# Install Poetry
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock ./

# Copy all files
COPY . .

# Install project dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi


# Run main.py when the container launches, allowing for additional arguments
ENTRYPOINT ["python", "main.py"]
