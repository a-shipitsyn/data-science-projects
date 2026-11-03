# Basic trading bot

A Python system for algorithmic crypto trading with two products that share one
strategy and risk core: a **backtester** with a Streamlit dashboard, and a **live
paper-trading signal bot** on the exchange's real-time feed. Because both run the exact
same Strategy → Risk pipeline, a strategy behaves the same in research and live.

Paper trading only: the bot logs signals, it doesn't place orders. The example strategy
loses money (see below).

> **Disclaimer.** This software is provided "as is", without warranty of any kind. It is not
> financial, investment or trading advice. Trading cryptocurrencies carries a high risk of
> losing money. If you use, modify or connect this code to a real account, you do so entirely
> at your own risk and you are solely responsible for any decisions, trades, losses or other
> consequences. The author accepts no liability for any loss or damage arising from its use.

## Architecture

```
                        +------------------------------------+
                        |           COINBASE API             |
                        |  (REST candles & WebSocket ticker) |
                        +------------------------------------+
                                     |            |
             [Historical candles]    |            |      [Live ticks]
             (REST, via ccxt)        |            |      (WebSocket)
                                     v            v
+------------------------------------------+    +------------------------------------------+
|           BACKTESTER                     |    |           LIVE BOT                       |
|  1. CSV of OHLCV candles                 |    |  1. Ticks -> 1-minute candles            |
|                                          |    |                                          |
|        +-----------------------------------------------------------------+               |
| ======>|                       SHARED CORE                               |<======        |
|        |  a. Strategy: raw signal from closed candles (BUY / SELL / HOLD) |               |
|        |  b. Risk Manager: position size, stop-loss, take-profit         |               |
|        |  c. Validated signal                                            |               |
|        +-----------------------------------------------------------------+               |
|                                          |    |                                          |
|  2. Simulated trades, fees, SL / TP      |    |  2. Paper position, fees, SL / TP        |
|  3. Metrics + Streamlit dashboard        |    |  3. Signal log (CSV)                     |
+------------------------------------------+    +------------------------------------------+
```

## How it works

The system has three blocks. Two of them — the backtester and the live bot — differ only in
where the prices come from; both hand the prices to the third, shared block, which makes
every trading decision.

**1. Core — the decision-making.** It answers two questions for every new candle: *should we
trade?* and *how much?* The **strategy** looks at recent candles and proposes BUY, SELL or
HOLD. The **risk manager** then accepts or rejects the proposal and sizes it: how much to buy
so that a stop-loss costs only a fixed share of capital, where to place the stop-loss and
take-profit, and whether trading is allowed at all (the kill switch stops new trades after
too large a daily loss or drawdown). The core knows nothing about files, exchanges or
time — it receives candles and returns a decision, which is what makes it reusable.

**2. Backtester — the research lab.** It answers *would this have worked in the past?* It
downloads historical candles, replays them one by one through the core as if they were
arriving live (the strategy never sees the future), and simulates the resulting trades with
fees, slippage and stop-losses. The outcome is a set of metrics — return, drawdown, Sharpe
ratio, compared with simply buying and holding — and a dashboard to explore them. The
optimizer on top of it searches for good parameters and checks them on data that was not
used to choose them, to tell real skill from luck.

**3. Live bot — the same strategy in real time.** It answers *what would the strategy do
right now?* It listens to the exchange's live price stream, builds one-minute candles from
individual trades, and passes each finished candle through the same core. Instead of sending
orders it keeps a simulated (paper) account and writes every decision to a log, so live
behaviour can be compared with the backtest. Around it sits what a long-running service
needs: reconnecting when the feed drops, a watchdog for a silent feed, saving its state to
survive restarts, notifications, and packaging as a container.

## Components

### Core — shared logic

The backtester and the live bot use the same strategy and risk code, so a strategy behaves
the same in testing and live.

| File | What it does |
|---|---|
| [core/models.py](core/models.py) | Defines `Signal`, the one message format everything passes around: time, asset, direction (BUY / SELL / HOLD), price, size, stop-loss, take-profit, strategy name, status (raw, validated or rejected) and the reason for an exit. `to_dict()` turns it into a row of the signal log. |
| [core/base_strategy.py](core/base_strategy.py) | The template every strategy follows: how many recent candles it needs (`warmup`) and how it turns candles into a raw signal (`generate_raw_signal`). `on_data()` is the one entry point the backtester and the bot call: it gets the raw signal and passes BUY and SELL through the risk manager. |
| [core/strategies/sma_crossover.py](core/strategies/sma_crossover.py) | The example strategy. It compares a fast and a slow moving average of the closing price: BUY when the fast one crosses above the slow one, SELL when it crosses below. It needs the slow window plus one candle, to compare the current and previous values. |
| [core/risk_manager.py](core/risk_manager.py) | Decides whether a signal may trade, and how big. Long-only: BUY only when holding nothing, SELL only when holding. Position size is chosen so that hitting the stop-loss loses a fixed share of capital (e.g. 2 %), capped so the position plus its fee fits the money available; stop-loss and take-profit at fixed percentages from the entry price. **Kill switch**: after a daily loss limit it blocks new buys until the next day (UTC), after a maximum drawdown it blocks them for good; the open position is closed when it trips. On by default in the live bot (5 % daily, 20 % drawdown), off by default in the backtester. Its state can be saved and restored. |

### Backtester — testing on history

| File | What it does |
|---|---|
| [backtester/data_downloader.py](backtester/data_downloader.py) | Downloads price candles (open, high, low, close, volume) from Coinbase through ccxt, no API keys needed. The exchange returns at most 300 candles per request, so it fetches page by page, removes duplicates and saves a CSV, then reports any missing candles. |
| [backtester/data_quality.py](backtester/data_quality.py) | Finds gaps in the data — where the time between candles is longer than usual — and reports how many candles are missing. Used by the downloader, backtester and optimizer. |
| [backtester/backtest_engine.py](backtester/backtest_engine.py) | Simulates trading candle by candle: (1) checks stop-loss and take-profit, filling at the candle's open if the price gapped through the level; (2) updates the kill switch with the account value; (3) gives the strategy only the candles up to now, so it can't see the future; (4) executes a validated signal at the close, with fees on both sides and optional slippage. It records each trade and the account value over time, and computes return, buy-and-hold return, trades, win rate, **max drawdown** and an annualised **Sharpe ratio**. It can trade from a given date onward, using earlier candles only to warm up the indicators (used by the optimizer). |
| [backtester/optimizer.py](backtester/optimizer.py) | Chooses parameters without fooling itself. **Grid search** backtests every fast / slow pair and ranks them by Sharpe. **Walk-forward** cuts the later part of the data into test windows; for each one it picks the best pair using only earlier data, then trades that window. The combined result of the windows is an honest out-of-sample estimate, compared with buy-and-hold and with the best pair chosen in hindsight. |
| [backtester/app.py](backtester/app.py) | A Streamlit web dashboard: set the data file, strategy windows, risk, stop-loss, take-profit, fee and slippage, run a backtest, and see the metrics, a candlestick chart with both moving averages, the equity curve with drawdown, and the trade list. |

### Live bot — paper trading on real-time prices

| File | What it does |
|---|---|
| [trading_bot/websocket_client.py](trading_bot/websocket_client.py) | Connects to Coinbase's public live feed, subscribes to price ticks for the asset and passes each tick to the bot. If the connection drops it reconnects, waiting a little longer each time (up to 60 s); the watchdog can force a reconnect. |
| [trading_bot/main_bot.py](trading_bot/main_bot.py) | The bot itself. **Start:** loads saved paper state, then fetches recent closed 1-minute candles so the indicators are ready. **Ticks into candles:** each tick updates the current minute's high, low, close and volume; when a minute ends the candle is closed and the strategy runs once, on finished candles only. **Paper trading:** a validated BUY or SELL updates the simulated capital and position, fees included; stop-loss and take-profit are checked on every tick. **Kill switch:** at each candle close the account value is checked; if a limit trips, the position is sold, a notification is sent and buying stops. **Watchdog:** every second it closes candles on time, so a quiet minute becomes a flat candle, and reconnects if no ticks arrive for 60 s. **State and shutdown:** capital, position, levels and kill-switch state are saved to `data/live_logs/paper_state.json` after every trade and when stopped by Ctrl+C or `docker stop`. |
| [trading_bot/signal_logger.py](trading_bot/signal_logger.py) | Appends each executed BUY or SELL to `data/live_logs/live_signals.csv` exactly once, with the reason for each exit. |
| [trading_bot/notifier.py](trading_bot/notifier.py) | Sends trades, the kill switch and a silent feed to Telegram if `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set; otherwise only logs them. It sends in a background thread, so a slow network never delays price processing. |

### Other files

| File | What it does |
|---|---|
| [tests/](tests/) | Tests on generated candles that need no internet: risk sizing and the kill switch, strategy signals, fees in profit, gap fills, slippage, metrics, missing candles, walk-forward windows; for the live bot one log entry per action, flat candles, volume, notifications, kill switch and restoring state. |
| [Dockerfile](Dockerfile) | Packages the live bot as a container that runs as an unprivileged user and keeps its log and state in a data volume; `docker stop` makes it save its state and exit cleanly. |
| [requirements.txt](requirements.txt) | ccxt (exchange data), pandas, plotly and streamlit (dashboard), websockets (live feed), pytest (tests). |

## Example result

SMA crossover on 4995 hourly BTC-USD candles (2026-04-06 – 2026-11-01; the exchange history
has one gap of 5 candles), 2 % risk per trade, 5 % stop-loss, 10 % take-profit, 0.5 % fee
per side.

Default parameters (10 / 50), whole period:

| Return | Buy & hold | Trades | Win rate | Max drawdown | Sharpe |
|---|---|---|---|---|---|
| −22.7 % | +11.0 % | 69 | 18.8 % | −23.5 % | −4.4 |

Walk-forward test — grid fast {5, 10, 20} × slow {30, 50, 100}, first half for training,
then 4 test windows of about 26 days, each with parameters chosen on all earlier data:

| Window | Chosen | Test return | Test Sharpe | Buy & hold |
|---|---|---|---|---|
| 07-20 – 08-15 | 20 / 50 | −4.6 % | −7.8 | −2.8 % |
| 08-15 – 09-10 | 20 / 50 | −0.1 % | −0.0 | +23.8 % |
| 09-10 – 10-06 | 10 / 100 | +0.5 % | +0.8 | +9.4 % |
| 10-06 – 11-01 | 10 / 100 | −6.8 % | −12.5 | −9.3 % |
| **combined** | | **−10.7 %** | | **+19.4 %** |

Even the best pair chosen on all data in hindsight loses (−16.6 %), and out of sample the
strategy trails buy-and-hold by 30 percentage points: a plain moving-average crossover on
hourly candles loses to fees and whipsaws. The framework is the point here — the strategy is a
placeholder to replace, and the walk-forward test is how a replacement should be judged.

## Project structure

```
basic-trading-bot/
├── README.md
├── requirements.txt
├── core/
│   ├── models.py
│   ├── base_strategy.py
│   ├── risk_manager.py
│   └── strategies/sma_crossover.py
├── backtester/
│   ├── data_downloader.py
│   ├── data_quality.py
│   ├── backtest_engine.py
│   ├── optimizer.py
│   └── app.py
├── trading_bot/
│   ├── websocket_client.py
│   ├── signal_logger.py
│   ├── notifier.py
│   └── main_bot.py
├── tests/
└── Dockerfile            # live bot as a container
```

Data files are created when the scripts run: `data/historical/` (candles) and
`data/live_logs/` (signal log, paper account state).

## Usage

Run everything from the project root (imports are `from core...`):

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests

# 1. download candles (public data, no API keys)
python -m backtester.data_downloader --symbol BTC/USD --timeframe 1h --limit 5000

# 2. backtest from the command line, or in the dashboard
python -m backtester.backtest_engine --fast 10 --slow 50 --slippage 0.0005
PYTHONPATH=. streamlit run backtester/app.py

# 3. choose parameters out of sample
python -m backtester.optimizer --folds 4 --fast 5 10 20 --slow 30 50 100

# 4. live paper-trading signal bot on 1-minute candles (Ctrl+C to stop)
python -m trading_bot.main_bot --asset BTC-USD --fast 10 --slow 50 --risk 0.02 --sl 0.05 --tp 0.10 \
    --max-daily-loss 0.05 --max-drawdown 0.20
```

The kill switch can be tested in the backtester too: `--max-daily-loss 0.05 --max-drawdown 0.2`.

Live bot as a container (log and state in a named volume; optional Telegram notifications):

```sh
docker build -t basic-trading-bot .
docker run -d --name bot -v bot-data:/app/data \
    -e TELEGRAM_BOT_TOKEN=<token> -e TELEGRAM_CHAT_ID=<chat id> \
    basic-trading-bot --asset BTC-USD --fast 10 --slow 50
docker logs -f bot        # docker stop bot: saves the state and exits
```

To add a strategy, subclass `BaseStrategy`, implement `warmup` and `generate_raw_signal`,
and pass it to the engine or the bot.
