FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY hyperliquid_bot.py .

CMD ["python", "hyperliquid_bot.py"]
