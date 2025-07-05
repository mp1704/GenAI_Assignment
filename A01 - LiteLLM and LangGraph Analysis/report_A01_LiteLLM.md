---
title: litellm_comprehensive_tutorial
---

# LiteLLM Comprehensive Tutorial

---

## Overview and Core Concepts

<details>
<summary>Understanding LiteLLM's Role in Modern AI Development</summary>

---

### What is LiteLLM?

- **Unified LLM interface** - single API for 100+ language models from different providers
- **OpenAI-compatible format** - standardized interface across all supported models
- **Provider abstraction** - switch between OpenAI, Anthropic, Google, Azure, and others seamlessly
- **Production-ready features** - built-in retry logic, fallbacks, caching, and monitoring

### Core Value Proposition

- **Eliminate vendor lock-in** - easily switch between different LLM providers
- **Reduce integration complexity** - learn one API instead of multiple provider-specific APIs
- **Cost optimization** - route requests to most cost-effective models dynamically
- **Reliability enhancement** - automatic fallbacks and retry mechanisms

### Key Features Overview

- **Model routing** - intelligent selection based on latency, cost, or availability
- **Unified API** - consistent request/response format regardless of provider
- **Built-in observability** - comprehensive logging and monitoring capabilities
- **Enterprise features** - budget management, rate limiting, and access controls

---

</details>

---

## Installation and Setup

<details>
<summary>Complete Installation Guide and Environment Configuration</summary>

---

### Basic Installation

```bash
# Core installation
pip install litellm

# With optional dependencies for specific providers
pip install litellm[proxy]  # For proxy server features
pip install litellm[redis]  # For Redis caching
pip install litellm[extra_image_models]  # For image generation models
```

### Environment Configuration

#### API Keys Setup

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="..."

# Google/Vertex AI
export GOOGLE_API_KEY="..."
export VERTEX_PROJECT="your-project-id"
export VERTEX_LOCATION="us-central1"

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2023-12-01-preview"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION_NAME="us-east-1"
```

#### Verification Setup

```python
import litellm
import os

# Test basic functionality
def verify_installation():
    try:
        # Simple test call
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=10
        )
        print(f"✅ LiteLLM working! Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Setup issue: {e}")
        return False

verify_installation()
```

---

</details>

---

## Basic Usage Patterns

<details>
<summary>Fundamental Operations and Common Implementation Patterns</summary>

---

### Simple Completion Requests

```python
import litellm

# Basic chat completion
response = litellm.completion(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Multi-Provider Examples

```python
# OpenAI
openai_response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello from OpenAI!"}]
)

# Anthropic Claude
anthropic_response = litellm.completion(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Hello from Anthropic!"}]
)

# Google Gemini
google_response = litellm.completion(
    model="gemini-pro",
    messages=[{"role": "user", "content": "Hello from Google!"}]
)

# Azure OpenAI
azure_response = litellm.completion(
    model="azure/gpt-4-deployment",  # Use your deployment name
    messages=[{"role": "user", "content": "Hello from Azure!"}]
)
```

### Streaming Responses

```python
import litellm

def stream_example():
    stream = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Write a short story about AI."}],
        stream=True
    )
    
    print("Streaming response:")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    print("\n")

stream_example()
```

### Error Handling Best Practices

```python
import litellm
from litellm import exceptions

def robust_completion(messages, model="gpt-3.5-turbo", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                timeout=30
            )
            return response
        
        except exceptions.RateLimitError:
            print(f"Rate limit hit, attempt {attempt + 1}")
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except exceptions.AuthenticationError:
            print("Authentication failed - check API keys")
            break
            
        except exceptions.APIConnectionError:
            print(f"Connection error, attempt {attempt + 1}")
            time.sleep(1)
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    return None
```

---

</details>

---

## Advanced Features

<details>
<summary>Router Configuration, Fallbacks, and Production-Ready Patterns</summary>

---

### LiteLLM Router Setup

```python
from litellm import Router

# Configure multiple models with failover
router = Router(
    model_list=[
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": os.getenv("OPENAI_API_KEY")
            },
            "rpm": 500,
            "tpm": 100000
        },
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "azure/gpt-4-deployment",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE")
            },
            "rpm": 300,
            "tpm": 80000
        },
        {
            "model_name": "claude-3",
            "litellm_params": {
                "model": "claude-3-sonnet-20240229",
                "api_key": os.getenv("ANTHROPIC_API_KEY")
            },
            "rpm": 400,
            "tpm": 90000
        }
    ],
    routing_strategy="least-busy",  # Options: simple-shuffle, least-busy, usage-based-routing
    set_verbose=True
)

# Use router for requests
response = router.completion(
    model="gpt-4",  # Will route to available GPT-4 endpoint
    messages=[{"role": "user", "content": "Complex reasoning task"}]
)
```

### Fallback Configuration

```python
import litellm

# Set up automatic fallbacks
litellm.set_verbose = True

# Primary to backup model fallbacks
litellm.fallbacks = [
    {"gpt-4": ["gpt-3.5-turbo", "claude-3-sonnet-20240229"]},
    {"claude-3-opus-20240229": ["gpt-4", "gpt-3.5-turbo"]}
]

# Context window fallbacks (for token limit errors)
litellm.context_window_fallbacks = [
    {"gpt-3.5-turbo": ["gpt-3.5-turbo-16k"]},
    {"gpt-4": ["gpt-4-32k", "claude-3-sonnet-20240229"]}
]

def fallback_example():
    try:
        response = litellm.completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Very long prompt..."}],
            fallbacks=["gpt-3.5-turbo", "claude-3-sonnet-20240229"]
        )
        return response
    except Exception as e:
        print(f"All fallbacks failed: {e}")
        return None
```

### Caching Implementation

```python
import litellm
from litellm.caching import Cache

# Redis caching setup
litellm.cache = Cache(
    type="redis",
    host="localhost",
    port=6379,
    password="your-password"
)

# In-memory caching (for development)
litellm.cache = Cache(type="local")

def cached_completion_example():
    # First call - will hit the API
    response1 = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is machine learning?"}],
        caching=True,
        ttl=300  # Cache for 5 minutes
    )
    
    # Second call - will use cache
    response2 = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is machine learning?"}],
        caching=True
    )
    
    print(f"Same content: {response1.choices[0].message.content == response2.choices[0].message.content}")
```

### Budget and Usage Tracking

```python
import litellm

# Set global budget limits
litellm.max_budget = 10.0  # $10 limit
litellm.budget_manager = litellm.BudgetManager(project_name="my-ai-app")

def track_usage_example():
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Expensive model call"}]
    )
    
    # Check current usage
    current_cost = litellm._current_cost
    print(f"Current spend: ${current_cost:.4f}")
    
    # Get detailed cost breakdown
    cost_breakdown = litellm.completion_cost(
        model="gpt-4",
        usage=response.usage
    )
    print(f"This call cost: ${cost_breakdown:.6f}")
```

---

</details>

---

## Function Calling and Tool Integration

<details>
<summary>Advanced Tool Usage and Function Calling Patterns</summary>

---

### Basic Function Calling

```python
import litellm
import json

# Define tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Function implementation
def get_weather(location, unit="celsius"):
    # Mock weather data
    return json.dumps({
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "condition": "sunny"
    })

def function_calling_example():
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        tool_choice="auto"
    )
    
    # Check if function was called
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "get_weather":
                result = get_weather(**function_args)
                print(f"Weather result: {result}")
    
    return response
```

### Multi-Step Tool Workflows

```python
import litellm
import requests
import json

class ToolWorkflow:
    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "summarize_text",
                    "description": "Summarize a piece of text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to summarize"},
                            "max_length": {"type": "integer", "description": "Max summary length"}
                        },
                        "required": ["text"]
                    }
                }
            }
        ]
    
    def search_web(self, query):
        # Mock web search
        return f"Search results for '{query}': Latest AI developments include..."
    
    def summarize_text(self, text, max_length=100):
        # Simple summarization (in production, use proper summarization)
        words = text.split()
        return " ".join(words[:max_length]) + "..." if len(words) > max_length else text
    
    def execute_workflow(self, user_query):
        messages = [{"role": "user", "content": user_query}]
        
        while True:
            response = litellm.completion(
                model="gpt-4",
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            
            messages.append(response.choices[0].message.model_dump())
            
            if not response.choices[0].message.tool_calls:
                # No more tool calls, return final response
                return response.choices[0].message.content
            
            # Execute tool calls
            for tool_call in response.choices[0].message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_web":
                    result = self.search_web(**function_args)
                elif function_name == "summarize_text":
                    result = self.summarize_text(**function_args)
                else:
                    result = f"Unknown function: {function_name}"
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

# Usage example
workflow = ToolWorkflow()
result = workflow.execute_workflow("Find recent AI news and summarize it")
print(result)
```

---

</details>

---

## Production Deployment Patterns

<details>
<summary>Enterprise-Grade Configuration and Monitoring Setup</summary>

---

### LiteLLM Proxy Server

```bash
# Install proxy dependencies
pip install litellm[proxy]

# Create config file
cat > litellm_config.yaml << EOF
model_list:
  - model_name: gpt-4
    litellm_params:
      model: gpt-4
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      mode: chat
      
  - model_name: claude-3
    litellm_params:
      model: claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      mode: chat

general_settings:
  master_key: "sk-1234567890"  # Your proxy access key
  database_url: "postgresql://user:pass@localhost:5432/litellm"
  
router_settings:
  routing_strategy: "usage-based-routing"
  model_group_alias:
    gpt-4: ["gpt-4", "claude-3"]
    
litellm_settings:
  telemetry: false
  success_callback: ["langsmith", "prometheus"]
  failure_callback: ["langsmith", "prometheus"]
EOF

# Start proxy server
litellm --config litellm_config.yaml --port 4000
```

### Client Configuration for Proxy

```python
import litellm

# Configure client to use proxy
litellm.api_base = "http://localhost:4000"
litellm.api_key = "sk-1234567890"  # Your master key

def proxy_client_example():
    response = litellm.completion(
        model="gpt-4",  # Will route through proxy
        messages=[{"role": "user", "content": "Hello via proxy!"}]
    )
    return response
```

### Monitoring and Observability

```python
import litellm
from litellm.integrations.langsmith import LangsmithLogger

# Set up comprehensive logging
litellm.success_callback = ["langsmith", "prometheus", "custom_logger"]
litellm.failure_callback = ["langsmith", "prometheus", "custom_logger"]

def custom_logger(kwargs, response_obj, start_time, end_time):
    """Custom logging function"""
    duration = (end_time - start_time).total_seconds()
    
    log_data = {
        "model": kwargs.get("model"),
        "duration": duration,
        "tokens": response_obj.usage.total_tokens if hasattr(response_obj, 'usage') else 0,
        "cost": kwargs.get("response_cost", 0),
        "timestamp": start_time.isoformat()
    }
    
    # Send to your monitoring system
    print(f"API Call Log: {log_data}")

# Register custom callback
litellm.callbacks = [custom_logger]

def monitored_completion():
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Monitored request"}],
        metadata={"user_id": "user123", "session_id": "session456"}
    )
    return response
```

### Load Balancing Configuration

```python
from litellm import Router
import os

# Production router with load balancing
production_router = Router(
    model_list=[
        # Primary OpenAI endpoints
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "gpt-4",
                "api_key": os.getenv("OPENAI_API_KEY_1")
            },
            "rpm": 500,
            "tpm": 100000
        },
        {
            "model_name": "gpt-4", 
            "litellm_params": {
                "model": "gpt-4",
                "api_key": os.getenv("OPENAI_API_KEY_2")
            },
            "rpm": 500,
            "tpm": 100000
        },
        # Azure backup
        {
            "model_name": "gpt-4",
            "litellm_params": {
                "model": "azure/gpt-4-deployment",
                "api_key": os.getenv("AZURE_API_KEY"),
                "api_base": os.getenv("AZURE_API_BASE")
            },
            "rpm": 300,
            "tpm": 80000
        }
    ],
    routing_strategy="least-busy",
    set_verbose=True,
    num_retries=3,
    timeout=30,
    fallbacks=[
        {"gpt-4": ["claude-3-sonnet-20240229"]}
    ]
)

# Health check endpoint
def health_check():
    try:
        response = production_router.completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return {"status": "healthy", "model": response.model}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

---

</details>

---

## Best Practices and Optimization

<details>
<summary>Performance Optimization and Production-Ready Guidelines</summary>

---

### Cost Optimization Strategies

```python
import litellm

class CostOptimizedLLM:
    def __init__(self):
        self.model_costs = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
        
    def estimate_cost(self, prompt, model, max_tokens=150):
        """Estimate request cost before making call"""
        input_tokens = len(prompt.split()) * 1.3  # Rough estimation
        output_tokens = max_tokens
        
        model_pricing = self.model_costs.get(model, self.model_costs["gpt-3.5-turbo"])
        
        cost = (input_tokens * model_pricing["input"] / 1000 + 
                output_tokens * model_pricing["output"] / 1000)
        return cost
    
    def smart_model_selection(self, prompt, complexity="medium", budget_limit=0.01):
        """Select model based on complexity and budget"""
        models_by_cost = [
            ("gpt-3.5-turbo", "low"),
            ("claude-3-sonnet", "medium"), 
            ("gpt-4", "high")
        ]
        
        for model, model_complexity in models_by_cost:
            estimated_cost = self.estimate_cost(prompt, model)
            
            if estimated_cost <= budget_limit:
                if complexity == "low" or model_complexity == complexity:
                    return model
        
        return "gpt-3.5-turbo"  # Fallback to cheapest
    
    def optimized_completion(self, prompt, complexity="medium", budget_limit=0.01):
        selected_model = self.smart_model_selection(prompt, complexity, budget_limit)
        
        response = litellm.completion(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        
        actual_cost = litellm.completion_cost(
            model=selected_model,
            usage=response.usage
        )
        
        return {
            "response": response,
            "model_used": selected_model,
            "cost": actual_cost
        }

# Usage
optimizer = CostOptimizedLLM()
result = optimizer.optimized_completion(
    "Simple question about Python",
    complexity="low",
    budget_limit=0.005
)
```

### Rate Limiting and Concurrency

```python
import asyncio
import litellm
from asyncio import Semaphore
import time

class RateLimitedLLM:
    def __init__(self, max_concurrent=5, rate_limit_rpm=100):
        self.semaphore = Semaphore(max_concurrent)
        self.rate_limit = rate_limit_rpm
        self.last_request_time = 0
        self.request_count = 0
        self.window_start = time.time()
    
    async def wait_for_rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Reset window if needed
        if current_time - self.window_start >= 60:
            self.request_count = 0
            self.window_start = current_time
        
        # Check if we need to wait
        if self.request_count >= self.rate_limit:
            wait_time = 60 - (current_time - self.window_start)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.window_start = time.time()
    
    async def completion(self, **kwargs):
        async with self.semaphore:
            await self.wait_for_rate_limit()
            
            self.request_count += 1
            response = await litellm.acompletion(**kwargs)
            return response
    
    async def batch_completions(self, requests):
        """Process multiple requests with rate limiting"""
        tasks = []
        for request in requests:
            task = self.completion(**request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# Usage example
async def rate_limited_example():
    llm = RateLimitedLLM(max_concurrent=3, rate_limit_rpm=60)
    
    requests = [
        {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": f"Question {i}"}]
        }
        for i in range(10)
    ]
    
    results = await llm.batch_completions(requests)
    return results
```

### Error Recovery and Resilience

```python
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

class ResilientLLM:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def robust_completion(self, **kwargs):
        """Completion with built-in retry logic"""
        try:
            response = litellm.completion(**kwargs)
            return response
            
        except litellm.exceptions.RateLimitError as e:
            self.logger.warning(f"Rate limit hit: {e}")
            raise  # Will trigger retry
            
        except litellm.exceptions.APIConnectionError as e:
            self.logger.warning(f"Connection error: {e}")
            raise  # Will trigger retry
            
        except litellm.exceptions.AuthenticationError as e:
            self.logger.error(f"Auth error: {e}")
            # Don't retry auth errors
            return None
            
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
    
    def completion_with_circuit_breaker(self, **kwargs):
        """Implement circuit breaker pattern"""
        try:
            return self.robust_completion(**kwargs)
        except Exception as e:
            # Log the failure and potentially switch to fallback service
            self.logger.error(f"All retries failed: {e}")
            return self.fallback_response(kwargs.get("messages", []))
    
    def fallback_response(self, messages):
        """Provide fallback when all else fails"""
        return {
            "choices": [{
                "message": {
                    "content": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                    "role": "assistant"
                }
            }]
        }

# Usage
resilient_llm = ResilientLLM()
response = resilient_llm.completion_with_circuit_breaker(
    model="gpt-4",
    messages=[{"role": "user", "content": "Important request"}]
)
```

---

</details> 