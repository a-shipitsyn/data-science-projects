FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core core
COPY backtester backtester
COPY trading_bot trading_bot

# run as an unprivileged user; signal log and paper state live in the data volume
RUN useradd --create-home bot && mkdir -p data/live_logs && chown -R bot data
USER bot
VOLUME /app/data
ENV PYTHONUNBUFFERED=1

# docker stop sends SIGTERM: the bot saves its state and exits
ENTRYPOINT ["python", "-m", "trading_bot.main_bot"]
