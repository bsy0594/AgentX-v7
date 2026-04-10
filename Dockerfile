FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src src
COPY start_participant.sh .

RUN pip install --no-cache-dir -e .
RUN chmod +x start_participant.sh

EXPOSE 9009

ENTRYPOINT ["bash", "start_participant.sh"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
