FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir -r requirements.txt

COPY *.py .

CMD ["python", "hyperliquid_bot.py"]
