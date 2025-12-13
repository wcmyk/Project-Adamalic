# 🤖 Autonomous 24/7 LILITH Deployment

**Run your AI agent 24/7 with autonomous decision-making and scheduled operations.**

Perfect for: Trading bots, research automation, monitoring systems, data pipelines

---

## 🎯 Architecture: Autonomous vs On-Demand

### **On-Demand (Traditional):**
```
User → API Request → Model → Response → User
```
- Model only runs when called
- Cheap (pay per use)
- No autonomous behavior

### **Autonomous (24/7):**
```
Schedule/Event → Agent → Decision → Action → Log
     ↑                                          |
     |__________ Continuous Loop _______________|
```
- Runs continuously
- Makes decisions independently
- Executes scheduled tasks
- Monitors and responds to events

---

## 💰 24/7 Hosting Options (Cost Comparison)

### **Option 1: Hugging Face Spaces (FREE but limited)**

```python
# Hugging Face Spaces runs 24/7 for FREE
# But: CPU only on free tier, limited compute

import gradio as gr
import schedule
import threading
from datetime import datetime

# Your LILITH model
from transformers import pipeline
agent = pipeline("text-generation", model="your-model")

# Autonomous task runner
def scheduled_task():
    """Runs every hour, autonomous decision making."""

    # Get market data, news, etc.
    market_data = fetch_market_data()

    # AI analyzes and decides
    decision = agent(f"Analyze this market data: {market_data}")

    # Execute action
    if "BUY" in decision:
        execute_trade(decision)

    log(f"[{datetime.now()}] Decision: {decision}")

# Schedule continuous operation
schedule.every().hour.do(scheduled_task)

# Run in background thread
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

threading.Thread(target=run_scheduler, daemon=True).start()

# Also expose API for manual queries
def chat(message):
    return agent(message)

gr.Interface(chat, "text", "text").launch()
```

**Cost:** $0 (FREE)
**CPU:** Limited (CPU only)
**RAM:** 16GB
**Storage:** 50GB
**Uptime:** 99%+ (auto-restarts)
**Limitations:**
- CPU only (slow for large models)
- No GPU access on free tier
- Limited to 50GB storage

**Best for:** Small models (<350M params), low-frequency tasks

---

### **Option 2: Modal.com (Best Value for Autonomous Agents)**

```python
# Modal: Serverless GPUs, only pay when running
# Perfect for intermittent autonomous tasks

import modal
from datetime import datetime

stub = modal.Stub("autonomous-lilith")

# Define your compute requirements
@stub.function(
    gpu="T4",  # or "A10G" for larger models
    schedule=modal.Period(hours=1),  # Run every hour
    secrets=[modal.Secret.from_name("trading-api-keys")],
)
def autonomous_trading_agent():
    """Runs every hour, analyzes markets, makes trades."""

    # Load model (cached after first run)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("your-lilith-model")

    # Fetch market data
    market_data = get_market_snapshot()

    # AI reasoning
    prompt = f"""You are a quantitative trading agent.

    Market data: {market_data}

    Analyze and decide:
    1. Should we enter any positions?
    2. Should we exit current positions?
    3. What's the risk level?

    Provide reasoning and specific actions."""

    decision = model.generate(prompt)

    # Execute trades
    if should_execute(decision):
        execute_trades(decision)

    # Log to database
    log_decision(datetime.now(), decision, market_data)

    return {"status": "completed", "decision": decision}

# Also expose as API for manual queries
@stub.webhook(method="POST")
def api_endpoint(request):
    query = request["query"]
    response = model.generate(query)
    return {"response": response}
```

**Cost Breakdown:**
- **Intermittent (hourly checks):** ~$3-15/month
  - 1 hour/day active = $0.10/day × 30 = $3/month
  - 24/7 monitoring (1 min/hour) = $5-10/month
- **Heavy use (continuous):** ~$50-150/month
  - T4 GPU: $0.20/hour × 720 hours = $144/month
  - A10G GPU: $0.60/hour × 720 hours = $432/month

**FREE tier:** 30 GPU hours/month included!

**Best for:** Scheduled tasks, event-driven, intermittent operation

---

### **Option 3: Railway.app or Render.com (Always-On APIs)**

```python
# FastAPI server running 24/7
# Responds to webhooks, scheduled tasks, API calls

from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
import torch

app = FastAPI()
scheduler = BackgroundScheduler()

# Load model once at startup
from LILITH import load_checkpoint, create_quant_agent

model, tokenizer, _ = load_checkpoint("phoenix-5b.pt")
agent = create_quant_agent(model, tokenizer)

# Autonomous scheduled task
def hourly_market_analysis():
    """Runs every hour automatically."""

    market_data = fetch_markets()
    analysis = agent.chat(f"Analyze: {market_data}")

    # Make autonomous decisions
    if "high_confidence_trade" in analysis.lower():
        execute_trade(analysis)

    store_analysis(analysis)

# Schedule tasks
scheduler.add_job(hourly_market_analysis, 'interval', hours=1)
scheduler.add_job(daily_portfolio_review, 'cron', hour=9, minute=0)
scheduler.add_job(risk_check, 'interval', minutes=15)

scheduler.start()

# Also expose API
@app.post("/chat")
async def chat_endpoint(query: str):
    response = agent.chat(query)
    return {"response": response}

@app.post("/webhook/market-event")
async def market_event_webhook(event: dict):
    """Triggered by external market events."""

    # AI analyzes event immediately
    analysis = agent.chat(f"Market event: {event}")

    # Autonomous response
    if requires_action(analysis):
        execute_immediate_action(analysis)

    return {"status": "processed"}
```

**Cost:**
- **Railway.app:** $5-20/month (512MB-2GB RAM)
- **Render.com:** $7-25/month (similar specs)
- **Fly.io:** $5-15/month

**Best for:**
- API-driven autonomous agents
- Webhook-triggered actions
- Continuous availability with moderate compute

---

### **Option 4: Self-Hosted (Raspberry Pi to Server)**

#### **Option 4A: Raspberry Pi 5 (Cheapest 24/7)**

```python
# Run Phoenix-5B quantized on Raspberry Pi!
# 4-bit model = 2.5GB, fits on 8GB Pi

# One-time hardware cost: $150
# Electricity: ~$2/month (5W × 24/7)

import torch
from LILITH import load_checkpoint

# Load 4-bit quantized model
model = load_checkpoint("phoenix-5b-4bit.pt")

# Autonomous agent loop
import schedule
import time

def autonomous_task():
    # Your logic here
    data = fetch_data_from_api()
    decision = model.generate(f"Analyze: {data}")
    execute_action(decision)

# Run 24/7
schedule.every().hour.do(autonomous_task)

while True:
    schedule.run_pending()
    time.sleep(60)
```

**Cost:**
- **Initial:** $150 (Raspberry Pi 5, 8GB + SD card + case)
- **Monthly:** ~$2 (electricity)
- **Annual:** $24 (just electricity!)

**Performance:**
- Phoenix-5B (4-bit): Works, ~10 tokens/sec
- Smaller models (<1B): Very good, ~30 tokens/sec

**Best for:** Home hobbyists, low-frequency trading, personal research

---

#### **Option 4B: Gaming PC / Old Server**

```python
# Use your existing PC/laptop 24/7
# Or buy used server: $200-500

# Dell R720 Server (used): $300
# - 128GB RAM
# - 24 CPU cores
# - Add GTX 1060 6GB: $100 used
# Total: $400 one-time

# Electricity: ~$15-30/month (200W × 24/7)
```

**Cost:**
- **Initial:** $400 (used server + GPU)
- **Monthly:** $20 (electricity)
- **Annual:** $240

**Performance:**
- Phoenix-5B: 30-50 tokens/sec (excellent)
- Can run multiple models simultaneously

**Best for:** Serious hobbyists, small trading firms, researchers

---

#### **Option 4C: Dedicated GPU Server (Vast.ai, RunPod)**

```bash
# Rent dedicated 24/7 server with GPU

# Vast.ai spot instance (RTX 3090)
# $0.20-0.40/hour × 720 hours = $144-288/month

# RunPod 24/7 (RTX A4000)
# $0.29/hour × 720 hours = $209/month
```

**Cost:** $150-300/month depending on GPU
**Performance:** Excellent, production-ready
**Best for:** Professional use, trading firms, high-frequency operations

---

## 🤖 Autonomous Operation Frameworks

### **Framework 1: Time-Based Scheduling**

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from LILITH import create_quant_agent

agent = create_quant_agent(model, tokenizer)
scheduler = BackgroundScheduler()

# Market open analysis
scheduler.add_job(
    market_open_analysis,
    CronTrigger(hour=9, minute=30, day_of_week='mon-fri'),
    args=[agent]
)

# Hourly updates
scheduler.add_job(
    hourly_portfolio_check,
    'interval',
    hours=1,
    args=[agent]
)

# Market close summary
scheduler.add_job(
    market_close_summary,
    CronTrigger(hour=16, minute=0, day_of_week='mon-fri'),
    args=[agent]
)

# Weekend deep research
scheduler.add_job(
    deep_research_task,
    CronTrigger(hour=10, minute=0, day_of_week='sat'),
    args=[agent]
)

scheduler.start()

# Keep alive
import time
while True:
    time.sleep(60)
```

---

### **Framework 2: Event-Driven (React to External Events)**

```python
import asyncio
from websockets import connect
from LILITH import create_quant_agent

agent = create_quant_agent(model, tokenizer)

async def monitor_market_events():
    """Listen to market websocket, react immediately."""

    async with connect("wss://market-data-feed.com") as ws:
        while True:
            # Receive market event
            event = await ws.recv()

            # AI analyzes immediately
            analysis = agent.chat(f"""
            Market event: {event}

            Assess:
            1. Impact on our portfolio
            2. Immediate action needed?
            3. Risk level
            """)

            # Autonomous decision
            if "URGENT" in analysis:
                await execute_immediate_action(analysis)
            else:
                await log_event(event, analysis)

# Run event loop
asyncio.run(monitor_market_events())
```

---

### **Framework 3: Multi-Agent System (Specialized Agents)**

```python
from LILITH import create_quant_agent
import threading

# Load Phoenix-5B with different expert routing
market_analyzer = create_quant_agent(model, tokenizer, expert="quant")
risk_manager = create_quant_agent(model, tokenizer, expert="safety")
executor = create_quant_agent(model, tokenizer, expert="planning")

class AutonomousTradingSystem:
    """Multi-agent system with checks and balances."""

    def __init__(self):
        self.market_analyzer = market_analyzer
        self.risk_manager = risk_manager
        self.executor = executor

    def autonomous_cycle(self):
        """Run every 15 minutes."""

        # Agent 1: Analyze markets
        market_data = fetch_market_data()
        opportunities = self.market_analyzer.chat(f"""
        Analyze this market data and identify opportunities:
        {market_data}
        """)

        # Agent 2: Risk assessment
        risk_analysis = self.risk_manager.chat(f"""
        Evaluate risk of these opportunities:
        {opportunities}

        Current portfolio: {get_portfolio()}
        """)

        # Agent 3: Execute if approved
        if risk_analysis["risk_level"] == "acceptable":
            execution_plan = self.executor.chat(f"""
            Create execution plan for:
            {opportunities}

            Risk constraints: {risk_analysis}
            """)

            if execution_plan["confidence"] > 0.8:
                execute_trades(execution_plan)

        # Log consensus
        log_decision(opportunities, risk_analysis, execution_plan)

# Run autonomous system
system = AutonomousTradingSystem()

import schedule
schedule.every(15).minutes.do(system.autonomous_cycle)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

### **Framework 4: Self-Improving Agent (Research Mode)**

```python
from LILITH import create_quant_agent
from LILITH.reasoning import ChainOfThought, SelfCritique

agent = create_quant_agent(model, tokenizer)
reasoner = ChainOfThought(agent.model)
critic = SelfCritique(agent.model)

class SelfImprovingResearcher:
    """Agent that learns from its mistakes autonomously."""

    def __init__(self):
        self.performance_history = []

    def daily_research_task(self):
        """Runs every day, learns from previous performance."""

        # Generate research question
        question = self.generate_research_question()

        # Research with chain-of-thought
        research = reasoner.reason(question)

        # Self-critique
        critique = critic.critique(research)

        # Improve based on critique
        improved = agent.chat(f"""
        Original research: {research}
        Critique: {critique}

        Provide improved analysis addressing the critiques.
        """)

        # Execute action based on improved analysis
        action = self.decide_action(improved)
        result = execute_action(action)

        # Learn from outcome
        self.performance_history.append({
            "question": question,
            "analysis": improved,
            "action": action,
            "result": result,
            "success": self.evaluate_success(result)
        })

        # Adjust strategy based on history
        if len(self.performance_history) >= 7:
            self.weekly_strategy_update()

    def weekly_strategy_update(self):
        """Self-improvement: analyze what worked, adjust approach."""

        recent_performance = self.performance_history[-7:]

        meta_analysis = agent.chat(f"""
        Review the last week of autonomous decisions:
        {recent_performance}

        What patterns led to success?
        What should be changed?
        Provide updated strategy.
        """)

        # Update system prompts/parameters based on learning
        self.update_strategy(meta_analysis)

# Run autonomous researcher
researcher = SelfImprovingResearcher()

import schedule
schedule.every().day.at("02:00").do(researcher.daily_research_task)
schedule.every().sunday.at("03:00").do(researcher.weekly_strategy_update)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

---

## 💰 Cost Comparison: 24/7 Operation

| Platform | Setup Cost | Monthly Cost | Annual Cost | Best For |
|----------|------------|--------------|-------------|----------|
| **HF Spaces Free** | $0 | $0 | $0 | Small models, low frequency |
| **Modal.com** | $0 | $5-50 | $60-600 | Intermittent tasks |
| **Railway/Render** | $0 | $15-30 | $180-360 | Always-on API |
| **Raspberry Pi** | $150 | $2 | $24 | Home hobbyist |
| **Used Server** | $400 | $20 | $240 | Serious hobbyist |
| **Cloud GPU 24/7** | $0 | $150-300 | $1,800-3,600 | Professional |

---

## 🎯 Recommended Setup by Use Case

### **Trading Bot (React to Markets 24/7)**

**Best:** Modal.com event-driven + webhook triggers

```python
# Only pay when bot is actively thinking/trading
# ~$10-30/month for typical trading frequency

@stub.function(gpu="T4")
@stub.webhook()
def market_event_handler(event):
    # Triggered by market movements
    # AI analyzes in 2-5 seconds
    # Returns trading decision
    pass
```

**Why:** Pay per use, instant response, no idle costs

---

### **Research Agent (Daily Deep Analysis)**

**Best:** Self-hosted Raspberry Pi or Modal scheduled

```python
# Option A: Raspberry Pi ($150 + $2/month)
# Runs locally, complete privacy

# Option B: Modal scheduled ($5-10/month)
# Better performance, no hardware
```

**Why:** Low frequency = low cost, either works great

---

### **Monitoring System (Continuous Scanning)**

**Best:** Railway/Render always-on API

```python
# $15/month for 24/7 availability
# Receives webhooks, monitors feeds
# Alerts you on important events
```

**Why:** Always ready, stable, affordable for continuous operation

---

### **High-Frequency Operations (Milliseconds Matter)**

**Best:** Dedicated cloud GPU or self-hosted

```python
# Need local instance for speed
# Cloud: $150-300/month
# Self-hosted: $400 + $20/month
```

**Why:** Latency critical, need dedicated resources

---

## 🚀 Quick Start: Deploy Autonomous Agent Today

### **30-Minute Setup (Modal.com - FREE tier)**

```python
# 1. Install Modal
pip install modal

# 2. Create autonomous_agent.py
import modal

stub = modal.Stub("my-autonomous-agent")

@stub.function(
    schedule=modal.Period(hours=1),  # Runs every hour
    gpu="T4",
)
def autonomous_task():
    from transformers import pipeline

    # Load your model
    agent = pipeline("text-generation",
                     model="mistralai/Mistral-7B-Instruct-v0.2")

    # Autonomous decision making
    market_data = fetch_data()  # Your data source
    decision = agent(f"Analyze: {market_data}")

    # Execute action
    if "BUY" in decision:
        execute_trade(decision)

    return decision

# 3. Deploy
modal deploy autonomous_agent.py

# Done! Runs every hour automatically
# First 30 GPU hours/month FREE
```

---

## 📊 Real-World Example: Autonomous Quant Agent

```python
"""
Complete autonomous trading system.
Runs 24/7, makes independent decisions, self-improves.
"""

import modal
from datetime import datetime

stub = modal.Stub("autonomous-quant")

@stub.function(
    gpu="T4",
    schedule=modal.Period(hours=1),
    secrets=[modal.Secret.from_name("trading-keys")],
)
def hourly_market_cycle():
    """Runs every hour, fully autonomous."""

    from transformers import AutoModelForCausalLM
    from LILITH.tools import create_quant_tools

    # Load Phoenix-5B (cached after first run)
    model = AutoModelForCausalLM.from_pretrained("your-phoenix-5b")
    tools = create_quant_tools()

    # 1. Gather market intelligence
    market_snapshot = {
        "prices": get_current_prices(),
        "volatility": calculate_volatility(),
        "portfolio": get_portfolio_status(),
        "news": get_market_news(),
    }

    # 2. AI analyzes with chain-of-thought
    analysis = model.generate(f"""
    You are an autonomous quantitative trading agent.

    Market Snapshot:
    {market_snapshot}

    Analyze step-by-step:
    1. What opportunities exist?
    2. What are the risks?
    3. How does this fit our portfolio?
    4. Should we take action?

    Provide reasoning and decision.
    """)

    # 3. Risk check
    if analysis["confidence"] > 0.75 and analysis["risk"] < 0.3:
        # Execute trade
        result = execute_trade(analysis["action"])

        # Log outcome for learning
        store_decision({
            "timestamp": datetime.now(),
            "analysis": analysis,
            "action": analysis["action"],
            "result": result,
        })

    # 4. Self-improvement: weekly review
    if datetime.now().weekday() == 6:  # Sunday
        review_week_performance()
        update_strategy_based_on_results()

    return {"status": "completed", "action": analysis.get("action", "HOLD")}

# Deploy and forget - runs autonomously
```

**Cost:** ~$15-25/month (T4 GPU, 1 hour/day active)
**Performance:** Claude-level analysis, independent operation
**ROI:** If it makes one good trade per month, pays for itself 100x over

---

## ✅ Next Steps

1. **Choose your hosting based on use case:**
   - Low frequency: Modal or Raspberry Pi
   - 24/7 API: Railway/Render
   - High performance: Dedicated GPU

2. **Implement autonomous framework:**
   - Time-based: Use APScheduler
   - Event-driven: Use webhooks + Modal
   - Multi-agent: Run multiple specialized instances

3. **Add monitoring:**
   - Log all decisions
   - Track performance
   - Alert on failures

4. **Self-improvement:**
   - Review weekly performance
   - Adjust strategies based on results
   - Continuous learning loop

---

**You can have an autonomous AI agent running 24/7 for as little as $2/month (Raspberry Pi) or $15/month (cloud with GPU).** 🚀
