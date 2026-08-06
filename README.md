# 🥇 XAU/USD AI Trading Bot — Production Edition

A production-grade algorithmic trading system for Gold (XAU/USD) featuring regime detection, Smart Money Concepts (SMC), risk management, and automated execution via MetaTrader 5 or Paper Trading.

## 🏗️ Architecture

```
gold_bot/
├── app.py                      # Streamlit dashboard entry point
├── src/
│   ├── constants.py            # Enums and constants
│   ├── exceptions.py           # Custom exceptions
│   ├── models.py               # Data models (dataclasses)
│   ├── config.py               # Centralized configuration (.env)
│   ├── logger.py               # Structured logging with rotation
│   ├── trading/
│   │   ├── broker_connector.py # Paper + MT5 brokers
│   │   ├── indicators.py       # Technical indicators
│   │   ├── strategies.py       # Trading strategies
│   │   ├── risk_manager.py     # Kelly, drawdown, daily guards
│   │   ├── regime_detector.py  # Market regime classification
│   │   ├── smc.py              # Smart Money Concepts
│   │   └── decision_engine.py  # Signal gating/confluence
│   ├── services/
│   │   ├── data_service.py     # OHLCV fetching with caching
│   │   ├── journal_service.py  # SQLite trade journal
│   │   ├── notification_service.py  # Telegram/Discord alerts
│   │   └── health_service.py   # Health monitoring
│   ├── backtesting/
│   │   ├── backtest_engine.py  # Backtest simulation
│   │   ├── monte_carlo.py      # Monte Carlo risk analysis
│   │   └── performance_report.py    # Performance analytics
│   ├── ui/
│   │   └── components.py       # Reusable Streamlit components
│   └── utils/
│       ├── validators.py       # Input validation
│       └── helpers.py          # Utility functions
├── tests/                      # Comprehensive pytest suite
├── Dockerfile                  # Production container
├── docker-compose.yml          # Multi-service orchestration
├── .github/workflows/ci.yml    # GitHub Actions CI/CD
├── requirements.txt
├── .env.example
└── pytest.ini
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url>
cd gold_bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Run

```bash
# Development
streamlit run app.py

# Production (Docker)
docker-compose up -d
```

## ⚙️ Configuration

All settings are controlled via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `BROKER` | `paper` or `mt5` | `paper` |
| `MT5_LOGIN` | MT5 account number | `0` |
| `MT5_PASSWORD` | MT5 password | `""` |
| `MT5_SERVER` | MT5 broker server | `""` |
| `MT5_LEVERAGE` | Account leverage | `100` |
| `RISK_BASE_PCT` | Base risk per trade | `1.0` |
| `RISK_MAX_PCT` | Max risk per trade | `2.0` |
| `RISK_DAILY_LOSS_LIMIT_PCT` | Daily loss circuit breaker | `5.0` |
| `RISK_USE_KELLY_SIZING` | Enable Kelly Criterion | `false` |
| `RISK_USE_AUTO_DRAWDOWN` | Auto-reduce risk on drawdown | `true` |
| `ANTHROPIC_API_KEY` | Claude AI key (optional) | `""` |
| `ALPHA_VANTAGE_API_KEY` | News API key (optional) | `""` |
| `TELEGRAM_BOT_TOKEN` | Telegram alerts (optional) | `""` |
| `DISCORD_WEBHOOK_URL` | Discord alerts (optional) | `""` |

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_risk_manager.py -v
```

## 🔒 Security

- **Never commit `.env`** — it contains credentials
- API keys loaded from environment, not session state
- Input validation on all user inputs
- No bare `except:` clauses — structured exception handling
- Rotating log files prevent disk exhaustion

## 📊 Features

- **Regime Detection**: Classifies market as trend/range/volatility
- **SMC Analysis**: Swing points, order blocks, FVGs, liquidity sweeps
- **Risk Management**: Kelly sizing, drawdown adjustment, daily loss limits
- **Paper Trading**: Realistic margin and lot sizing simulation
- **MT5 Integration**: Auto-reconnect, symbol resolution, full order management
- **SQLite Journal**: Persistent, queryable trade history
- **Notifications**: Telegram/Discord alerts for trades and errors
- **Backtesting**: Strategy validation with regime-aware simulation
- **Monte Carlo**: Risk of ruin and confidence interval analysis
- **Health Monitoring**: Automatic broker health checks with alerts

## 🐳 Docker

```bash
# Build and run
docker-compose up --build -d

# View logs
docker-compose logs -f gold-bot

# Stop
docker-compose down
```

## 🔄 CI/CD

GitHub Actions pipeline:
1. Lint with `flake8`
2. Format check with `black`
3. Run `pytest` with coverage
4. Build and test Docker image

## 📈 Performance

- `lru_cache` on SMC calculations and yfinance data
- SQLite indexes for fast journal queries
- Minimal DataFrame copies in indicators
- Background health monitoring thread

## 📝 License

MIT License — use at your own risk. Trading involves substantial risk of loss.

## ⚠️ Disclaimer

This software is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly in paper trading before live deployment.
