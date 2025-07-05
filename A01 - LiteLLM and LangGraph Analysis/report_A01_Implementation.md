---
title: implementation_examples
---

# Implementation Examples: LiteLLM & LangGraph

---

## Quick Start Examples

<details>
<summary>Essential Code Patterns for Immediate Use</summary>

---

### LiteLLM Quick Start

```python
import litellm
import os

# Basic setup
os.environ["OPENAI_API_KEY"] = "your-key"
os.environ["ANTHROPIC_API_KEY"] = "your-key"

# Simple completion with fallbacks
def quick_completion(prompt, model="gpt-3.5-turbo"):
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        fallbacks=["claude-3-sonnet-20240229", "gpt-4"],
        max_tokens=150
    )
    return response.choices[0].message.content

# Multi-provider comparison
def compare_providers(prompt):
    models = ["gpt-3.5-turbo", "claude-3-sonnet-20240229"]
    results = {}
    
    for model in models:
        try:
            result = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            cost = litellm.completion_cost(model=model, usage=result.usage)
            results[model] = {
                "response": result.choices[0].message.content,
                "cost": cost,
                "tokens": result.usage.total_tokens
            }
        except Exception as e:
            results[model] = {"error": str(e)}
    
    return results

# Usage
print(quick_completion("Explain AI in one sentence"))
print(compare_providers("What is quantum computing?"))
```

### LangGraph Quick Start

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated, List
import operator

# Define state
class SimpleState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    step_count: int

# Processing node
def process_node(state: SimpleState):
    user_input = state["messages"][-1].content
    response = f"Processed: {user_input} (Step {state['step_count'] + 1})"
    
    return {
        "messages": [AIMessage(content=response)],
        "step_count": state["step_count"] + 1
    }

# Create workflow
def create_simple_workflow():
    workflow = StateGraph(SimpleState)
    workflow.add_node("process", process_node)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    return workflow.compile()

# Usage
agent = create_simple_workflow()
result = agent.invoke({
    "messages": [HumanMessage(content="Hello!")],
    "step_count": 0
})

print(f"Response: {result['messages'][-1].content}")
```

---

</details>

---

## Real-World Applications

<details>
<summary>Customer Support Agent with LangGraph + LiteLLM</summary>

---

### Complete Customer Support System

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import litellm
import json

class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    issue_category: str
    urgency_level: str
    customer_tier: str
    resolution_status: str

# Categorize customer issue
def categorize_issue_node(state: SupportState):
    user_message = state["messages"][-1].content
    
    prompt = f"""
    Categorize this support request:
    "{user_message}"
    
    Return JSON with:
    - category: technical/billing/account/general
    - urgency: low/medium/high/critical
    - reasoning: brief explanation
    """
    
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        fallbacks=["claude-3-sonnet-20240229"],
        max_tokens=200
    )
    
    try:
        result = json.loads(response.choices[0].message.content)
        category = result.get("category", "general")
        urgency = result.get("urgency", "medium")
    except:
        category = "general"
        urgency = "medium"
    
    return {
        "issue_category": category,
        "urgency_level": urgency,
        "messages": [AIMessage(content=f"Categorized as {category} issue with {urgency} priority")]
    }

# Generate response based on category
def generate_response_node(state: SupportState):
    user_message = state["messages"][0].content
    category = state["issue_category"]
    urgency = state["urgency_level"]
    
    # Select model based on urgency
    model = "gpt-4" if urgency in ["high", "critical"] else "gpt-3.5-turbo"
    
    prompt = f"""
    You are a customer support agent responding to a {category} issue with {urgency} urgency.
    
    Customer message: {user_message}
    
    Provide a helpful, professional response with specific next steps.
    """
    
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        fallbacks=["gpt-3.5-turbo"],
        max_tokens=300
    )
    
    return {
        "messages": [AIMessage(content=response.choices[0].message.content)],
        "resolution_status": "responded"
    }

# Build support workflow
def create_support_agent():
    workflow = StateGraph(SupportState)
    
    workflow.add_node("categorize", categorize_issue_node)
    workflow.add_node("respond", generate_response_node)
    
    workflow.set_entry_point("categorize")
    workflow.add_edge("categorize", "respond")
    workflow.add_edge("respond", END)
    
    return workflow.compile()

# Test the agent
support_agent = create_support_agent()

test_cases = [
    "My API is returning 500 errors since this morning!",
    "How do I update my billing information?",
    "I can't log into my account"
]

for case in test_cases:
    result = support_agent.invoke({
        "messages": [HumanMessage(content=case)],
        "issue_category": "",
        "urgency_level": "",
        "customer_tier": "standard",
        "resolution_status": ""
    })
    
    print(f"\nIssue: {case}")
    print(f"Category: {result['issue_category']}")
    print(f"Urgency: {result['urgency_level']}")
    print(f"Response: {result['messages'][-1].content}")
```

---

</details>

---

## Cost Optimization Patterns

<details>
<summary>Smart Model Selection and Budget Management</summary>

---

### Intelligent Cost Optimizer

```python
import litellm
from datetime import datetime

class CostOptimizedAgent:
    def __init__(self, daily_budget=10.0):
        self.daily_budget = daily_budget
        self.daily_spend = 0.0
        
        # Model cost mapping (per 1K tokens)
        self.model_costs = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
        
        # Task complexity to model mapping
        self.task_models = {
            "simple": ["gpt-3.5-turbo", "claude-3-haiku-20240307"],
            "medium": ["gpt-3.5-turbo", "claude-3-sonnet-20240229"],
            "complex": ["gpt-4", "claude-3-sonnet-20240229"],
            "creative": ["gpt-4", "claude-3-sonnet-20240229"]
        }
    
    def analyze_task_complexity(self, prompt: str) -> str:
        """Determine task complexity from prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["creative", "story", "poem"]):
            return "creative"
        elif any(word in prompt_lower for word in ["analyze", "compare", "detailed"]):
            return "complex"
        elif any(word in prompt_lower for word in ["summarize", "list", "what is"]):
            return "simple"
        else:
            return "medium"
    
    def estimate_cost(self, model: str, prompt: str, max_tokens: int = 150) -> float:
        """Estimate request cost."""
        costs = self.model_costs.get(model, self.model_costs["gpt-3.5-turbo"])
        
        # Rough token estimation
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = max_tokens
        
        return (input_tokens * costs["input"] / 1000) + (output_tokens * costs["output"] / 1000)
    
    def select_optimal_model(self, prompt: str, max_cost: float = None) -> str:
        """Select best model within budget."""
        complexity = self.analyze_task_complexity(prompt)
        candidates = self.task_models[complexity]
        
        if max_cost is None:
            max_cost = self.daily_budget - self.daily_spend
        
        # Find cheapest suitable model
        for model in candidates:
            if self.estimate_cost(model, prompt) <= max_cost:
                return model
        
        return "claude-3-haiku-20240307"  # Cheapest fallback
    
    def optimized_completion(self, prompt: str, **kwargs):
        """Execute completion with cost optimization."""
        selected_model = self.select_optimal_model(prompt)
        estimated_cost = self.estimate_cost(selected_model, prompt)
        
        # Budget check
        if self.daily_spend + estimated_cost > self.daily_budget:
            return {
                "error": "Daily budget exceeded",
                "budget_remaining": self.daily_budget - self.daily_spend
            }
        
        try:
            response = litellm.completion(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                fallbacks=["gpt-3.5-turbo"],
                **kwargs
            )
            
            actual_cost = litellm.completion_cost(model=selected_model, usage=response.usage)
            self.daily_spend += actual_cost
            
            return {
                "response": response.choices[0].message.content,
                "model_used": selected_model,
                "cost": actual_cost,
                "budget_remaining": self.daily_budget - self.daily_spend,
                "complexity": self.analyze_task_complexity(prompt)
            }
            
        except Exception as e:
            return {"error": str(e)}

# Usage example
optimizer = CostOptimizedAgent(daily_budget=5.0)

test_prompts = [
    "What is AI?",  # Simple
    "Write a creative story about robots",  # Creative
    "Analyze the impact of climate change on agriculture"  # Complex
]

for prompt in test_prompts:
    result = optimizer.optimized_completion(prompt, max_tokens=200)
    
    if "error" not in result:
        print(f"\nPrompt: {prompt}")
        print(f"Model: {result['model_used']}")
        print(f"Complexity: {result['complexity']}")
        print(f"Cost: ${result['cost']:.6f}")
        print(f"Budget remaining: ${result['budget_remaining']:.6f}")
        print(f"Response: {result['response'][:100]}...")
    else:
        print(f"Error: {result['error']}")
```

---

</details>

---

## Production Deployment

<details>
<summary>FastAPI Service with Monitoring</summary>

---

### Complete Production Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time
import logging

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None
    max_cost: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    model_used: str
    cost: float
    processing_time: float

# Service implementation
class AIService:
    def __init__(self):
        self.optimizer = CostOptimizedAgent()
        self.request_count = 0
        self.total_cost = 0.0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        start_time = time.time()
        
        self.logger.info(f"Processing request from user {request.user_id}")
        
        try:
            result = self.optimizer.optimized_completion(
                request.message,
                max_tokens=300
            )
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            processing_time = time.time() - start_time
            self.request_count += 1
            self.total_cost += result["cost"]
            
            return ChatResponse(
                response=result["response"],
                model_used=result["model_used"],
                cost=result["cost"],
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Request failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# FastAPI app
app = FastAPI(title="AI Service", version="1.0.0")
ai_service = AIService()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    return await ai_service.process_chat(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "requests": ai_service.request_count}

@app.get("/metrics")
async def get_metrics():
    return {
        "requests_processed": ai_service.request_count,
        "total_cost": ai_service.total_cost,
        "average_cost": ai_service.total_cost / max(ai_service.request_count, 1)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

</details>

---

## Integration Patterns

<details>
<summary>Hybrid Architecture Examples</summary>

---

### LiteLLM within LangGraph Nodes

```python
from langgraph.graph import StateGraph, END
import litellm

class HybridState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_model: str
    total_cost: float

def smart_llm_node(state: HybridState):
    """LangGraph node using LiteLLM for optimal model selection."""
    user_input = state["messages"][-1].content
    
    # Analyze complexity
    complexity = "complex" if len(user_input.split()) > 50 else "simple"
    model = "gpt-4" if complexity == "complex" else "gpt-3.5-turbo"
    
    # Use LiteLLM with fallbacks
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": user_input}],
        fallbacks=["gpt-3.5-turbo", "claude-3-sonnet-20240229"]
    )
    
    cost = litellm.completion_cost(model=model, usage=response.usage)
    
    return {
        "messages": [AIMessage(content=response.choices[0].message.content)],
        "current_model": model,
        "total_cost": state.get("total_cost", 0) + cost
    }

# Create hybrid workflow
workflow = StateGraph(HybridState)
workflow.add_node("smart_llm", smart_llm_node)
workflow.set_entry_point("smart_llm")
workflow.add_edge("smart_llm", END)

hybrid_agent = workflow.compile()

# Test
result = hybrid_agent.invoke({
    "messages": [HumanMessage(content="Explain quantum computing in detail")],
    "current_model": "",
    "total_cost": 0.0
})

print(f"Model used: {result['current_model']}")
print(f"Cost: ${result['total_cost']:.6f}")
print(f"Response: {result['messages'][-1].content[:200]}...")
```

### Router-Based Multi-Step Workflow

```python
class IntelligentRouter:
    """Route workflow steps to optimal models."""
    
    def __init__(self):
        self.step_configs = {
            "planning": {"model": "gpt-4", "temperature": 0.1},
            "research": {"model": "claude-3-sonnet-20240229", "temperature": 0.3},
            "synthesis": {"model": "gpt-3.5-turbo", "temperature": 0.7}
        }
    
    def execute_step(self, step_name: str, messages: List):
        config = self.step_configs[step_name]
        
        response = litellm.completion(
            model=config["model"],
            messages=messages,
            temperature=config["temperature"],
            fallbacks=["gpt-3.5-turbo"]
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": config["model"],
            "cost": litellm.completion_cost(model=config["model"], usage=response.usage)
        }

# Use in LangGraph workflow
router = IntelligentRouter()

def planning_node(state):
    result = router.execute_step("planning", state["messages"])
    return {"messages": [AIMessage(content=result["content"])]}

def research_node(state):
    result = router.execute_step("research", state["messages"])
    return {"messages": [AIMessage(content=result["content"])]}

def synthesis_node(state):
    result = router.execute_step("synthesis", state["messages"])
    return {"messages": [AIMessage(content=result["content"])]}

# Build complete workflow
workflow = StateGraph(HybridState)
workflow.add_node("planning", planning_node)
workflow.add_node("research", research_node)
workflow.add_node("synthesis", synthesis_node)

workflow.set_entry_point("planning")
workflow.add_edge("planning", "research")
workflow.add_edge("research", "synthesis")
workflow.add_edge("synthesis", END)

routed_agent = workflow.compile()

# Test multi-step workflow
result = routed_agent.invoke({
    "messages": [HumanMessage(content="Create a comprehensive analysis of renewable energy trends")],
    "current_model": "",
    "total_cost": 0.0
})

print("=== Multi-Step Workflow Results ===")
for i, msg in enumerate(result["messages"]):
    if isinstance(msg, AIMessage):
        print(f"Step {i}: {msg.content[:150]}...")
```

---

</details> 