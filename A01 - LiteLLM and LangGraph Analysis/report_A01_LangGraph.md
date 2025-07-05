---
title: langgraph_comprehensive_tutorial
---

# LangGraph Comprehensive Tutorial

---

## Core Concepts and Architecture

<details>
<summary>Understanding LangGraph's Graph-Based Agent Framework</summary>

---

### What is LangGraph?

- **Graph-based orchestration** - build complex workflows as connected nodes and edges
- **Stateful execution** - maintain context and data across multiple interaction steps
- **Cyclic workflows** - support loops, conditional branching, and dynamic routing
- **Production-ready agents** - built-in persistence, human-in-the-loop, and error handling

### Key Architectural Components

#### Nodes
- **Processing units** - individual functions that perform specific tasks
- **State modification** - each node can read and update the shared state
- **Tool integration** - nodes can call external APIs, databases, or LangChain tools
- **Decision points** - nodes can determine next steps based on results

#### Edges  
- **Flow control** - define how execution moves between nodes
- **Conditional logic** - edges can be conditional based on state or outputs
- **Parallel execution** - multiple edges can enable concurrent processing
- **Loop support** - edges can create cycles for iterative processing

#### State Management
- **Shared context** - centralized state accessible to all nodes
- **Type safety** - state schemas ensure data consistency
- **Persistence** - state can be saved and restored across sessions
- **Versioning** - track state changes for debugging and rollback

### LangGraph vs Traditional Chains

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| Flow Type | Linear chains | Graph-based workflows |
| State Handling | Passed through chain | Centralized state management |
| Conditionals | Limited | Built-in conditional routing |
| Loops | Not supported | Native loop support |
| Human-in-Loop | Manual implementation | Built-in checkpoints |
| Persistence | External setup required | Built-in state persistence |

---

</details>

---

## Installation and Basic Setup

<details>
<summary>Environment Configuration and Initial Project Setup</summary>

---

### Installation

```bash
# Core LangGraph installation
pip install langgraph

# With additional dependencies
pip install langgraph[dev]  # Development tools
pip install langchain langchain-openai  # LangChain integration
pip install langsmith  # Observability (optional)
```

### Environment Configuration

```bash
# API Keys for LLM providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="..."

# LangSmith for observability (optional)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="..."
export LANGCHAIN_PROJECT="langgraph-tutorial"
```

### Basic Project Structure

```python
import operator
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Define the state structure
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    data: dict

# Verify setup
def verify_installation():
    print("✅ LangGraph installed successfully!")
    
    # Create a simple graph
    workflow = StateGraph(AgentState)
    workflow.add_node("start", lambda state: {"current_step": "initialized"})
    workflow.set_entry_point("start")
    workflow.add_edge("start", END)
    
    app = workflow.compile()
    result = app.invoke({"messages": [], "current_step": "", "data": {}})
    print(f"✅ Basic workflow test: {result['current_step']}")

verify_installation()
```

---

</details>

---

## Building Your First Agent

<details>
<summary>Step-by-Step Agent Construction with Practical Examples</summary>

---

### Simple Conversational Agent

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated, List
import operator

# Define state
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define the chat node
def chat_node(state: ConversationState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Build the graph
def create_simple_agent():
    workflow = StateGraph(ConversationState)
    
    # Add nodes
    workflow.add_node("chat", chat_node)
    
    # Set entry point and edges
    workflow.set_entry_point("chat")
    workflow.add_edge("chat", END)
    
    return workflow.compile()

# Usage example
simple_agent = create_simple_agent()

def test_simple_agent():
    result = simple_agent.invoke({
        "messages": [HumanMessage(content="Hello! How are you?")]
    })
    
    print("Conversation:")
    for msg in result["messages"]:
        print(f"{msg.__class__.__name__}: {msg.content}")

test_simple_agent()
```

### Agent with Tools and Decision Making

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json

# Define tools
@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Mock weather API
    return f"The weather in {location} is sunny and 72°F"

@tool  
def search_web(query: str) -> str:
    """Search the web for information."""
    # Mock search API
    return f"Search results for '{query}': Found relevant information..."

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)  # Note: In production, use safer evaluation
        return f"Result: {result}"
    except:
        return "Error: Invalid expression"

# Set up tools
tools = [get_weather, search_web, calculator]
tool_executor = ToolExecutor(tools)

# Enhanced state for tool usage
class ToolAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    pending_tool_calls: List[dict]

# Initialize LLM with tools
llm_with_tools = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(tools)

def agent_node(state: ToolAgentState):
    """Main agent reasoning node."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Check if tools need to be called
    if response.tool_calls:
        return {
            "messages": [response],
            "pending_tool_calls": response.tool_calls
        }
    else:
        return {"messages": [response]}

def tool_node(state: ToolAgentState):
    """Execute tools and return results."""
    tool_calls = state["pending_tool_calls"]
    tool_messages = []
    
    for tool_call in tool_calls:
        # Create tool invocation
        action = ToolInvocation(
            tool=tool_call["name"],
            tool_input=tool_call["args"]
        )
        
        # Execute tool
        result = tool_executor.invoke(action)
        
        # Create tool message
        tool_message = {
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call["id"]
        }
        tool_messages.append(tool_message)
    
    return {"messages": tool_messages}

def should_continue(state: ToolAgentState):
    """Decide whether to continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message has tool calls, continue to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"

# Build the agent workflow
def create_tool_agent():
    workflow = StateGraph(ToolAgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tools always go back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()

# Usage example
tool_agent = create_tool_agent()

def test_tool_agent():
    # Test weather query
    result1 = tool_agent.invoke({
        "messages": [HumanMessage(content="What's the weather in New York?")],
        "pending_tool_calls": []
    })
    
    print("\n=== Weather Query ===")
    for msg in result1["messages"]:
        if hasattr(msg, 'content'):
            print(f"{msg.__class__.__name__}: {msg.content}")
    
    # Test calculation
    result2 = tool_agent.invoke({
        "messages": [HumanMessage(content="Calculate 15 * 24 + 7")],
        "pending_tool_calls": []
    })
    
    print("\n=== Calculation Query ===")
    for msg in result2["messages"]:
        if hasattr(msg, 'content'):
            print(f"{msg.__class__.__name__}: {msg.content}")

test_tool_agent()
```

---

</details>

---

## Advanced Workflow Patterns

<details>
<summary>Complex State Management and Multi-Step Processes</summary>

---

### Multi-Stage Research Agent

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated, List, Dict
import operator
import json

# Complex state for research workflow
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    research_query: str
    search_results: List[Dict]
    analysis_notes: List[str]
    final_report: str
    current_stage: str
    iteration_count: int

llm = ChatOpenAI(model="gpt-4", temperature=0.1)

def query_analysis_node(state: ResearchState):
    """Analyze the user query and plan research approach."""
    query = state["messages"][-1].content
    
    analysis_prompt = f"""
    Analyze this research query and create a research plan:
    Query: {query}
    
    Provide:
    1. Key topics to research
    2. Specific questions to answer
    3. Research approach
    
    Format as JSON with keys: topics, questions, approach
    """
    
    response = llm.invoke([SystemMessage(content=analysis_prompt)])
    
    try:
        analysis = json.loads(response.content)
    except:
        analysis = {"topics": [query], "questions": [query], "approach": "general"}
    
    return {
        "research_query": query,
        "analysis_notes": [f"Research plan: {analysis}"],
        "current_stage": "search",
        "iteration_count": 0
    }

def search_node(state: ResearchState):
    """Simulate web search for information."""
    query = state["research_query"]
    
    # Mock search results
    mock_results = [
        {
            "title": f"Research on {query} - Academic Paper",
            "content": f"Detailed analysis of {query} with key findings...",
            "source": "academic_journal",
            "relevance": 0.9
        },
        {
            "title": f"{query} Industry Report 2024",
            "content": f"Latest trends and developments in {query}...",
            "source": "industry_report", 
            "relevance": 0.8
        },
        {
            "title": f"Case Study: {query} Implementation",
            "content": f"Real-world examples of {query} applications...",
            "source": "case_study",
            "relevance": 0.7
        }
    ]
    
    return {
        "search_results": mock_results,
        "current_stage": "analysis",
        "analysis_notes": state["analysis_notes"] + ["Search completed"]
    }

def analysis_node(state: ResearchState):
    """Analyze search results and extract insights."""
    results = state["search_results"]
    query = state["research_query"]
    
    analysis_prompt = f"""
    Analyze these search results for the query: {query}
    
    Results:
    {json.dumps(results, indent=2)}
    
    Provide:
    1. Key insights
    2. Important findings  
    3. Areas needing more research
    4. Preliminary conclusions
    """
    
    response = llm.invoke([SystemMessage(content=analysis_prompt)])
    
    return {
        "analysis_notes": state["analysis_notes"] + [response.content],
        "current_stage": "synthesis",
        "iteration_count": state["iteration_count"] + 1
    }

def synthesis_node(state: ResearchState):
    """Synthesize findings into final report."""
    query = state["research_query"]
    notes = state["analysis_notes"]
    results = state["search_results"]
    
    synthesis_prompt = f"""
    Create a comprehensive research report on: {query}
    
    Based on:
    Analysis Notes: {notes}
    Search Results: {json.dumps(results, indent=2)}
    
    Structure the report with:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Conclusions and Recommendations
    5. Sources
    """
    
    response = llm.invoke([SystemMessage(content=synthesis_prompt)])
    
    return {
        "final_report": response.content,
        "current_stage": "complete",
        "messages": [AIMessage(content=f"Research complete! Here's my comprehensive report:\n\n{response.content}")]
    }

def should_continue_research(state: ResearchState):
    """Determine next step in research process."""
    stage = state["current_stage"]
    iteration = state["iteration_count"]
    
    if stage == "search":
        return "analysis"
    elif stage == "analysis":
        # Could add logic to do more searches if needed
        if iteration < 2:  # Allow for iterative research
            return "synthesis"
        else:
            return "search"  # Go back for more data
    elif stage == "synthesis":
        return "complete"
    else:
        return "complete"

# Build research agent
def create_research_agent():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("search", search_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Set entry point
    workflow.set_entry_point("query_analysis")
    
    # Add edges
    workflow.add_edge("query_analysis", "search")
    
    # Conditional routing from analysis
    workflow.add_conditional_edges(
        "analysis",
        should_continue_research,
        {
            "synthesis": "synthesis",
            "search": "search",
            "complete": "synthesis"
        }
    )
    
    workflow.add_edge("search", "analysis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

# Usage example
research_agent = create_research_agent()

def test_research_agent():
    result = research_agent.invoke({
        "messages": [HumanMessage(content="Research the impact of AI on software development productivity")],
        "research_query": "",
        "search_results": [],
        "analysis_notes": [],
        "final_report": "",
        "current_stage": "start",
        "iteration_count": 0
    })
    
    print("=== Research Agent Results ===")
    print(f"Stage: {result['current_stage']}")
    print(f"Iterations: {result['iteration_count']}")
    print(f"\nFinal Report Preview:")
    print(result['final_report'][:500] + "..." if len(result['final_report']) > 500 else result['final_report'])

test_research_agent()
```

### Human-in-the-Loop Workflow

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, List
import operator

# State for human-in-the-loop workflow
class HumanLoopState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    pending_approval: str
    human_feedback: str
    iteration_count: int
    workflow_stage: str

def draft_content_node(state: HumanLoopState):
    """Create initial content draft."""
    user_request = state["messages"][-1].content
    
    draft_prompt = f"""
    Create a draft response for this request: {user_request}
    
    Make it comprehensive but mark areas where human review might be needed.
    """
    
    response = llm.invoke([SystemMessage(content=draft_prompt)])
    
    return {
        "pending_approval": response.content,
        "workflow_stage": "awaiting_review",
        "messages": [AIMessage(content="I've created a draft. Please review it.")]
    }

def human_review_node(state: HumanLoopState):
    """Wait for human review and feedback."""
    draft = state["pending_approval"]
    
    print("\n" + "="*50)
    print("HUMAN REVIEW REQUIRED")
    print("="*50)
    print("Draft content:")
    print(draft)
    print("\n" + "-"*30)
    
    # In production, this would integrate with a UI or messaging system
    feedback = input("Enter your feedback (or 'approve' to accept): ")
    
    if feedback.lower() == 'approve':
        return {
            "human_feedback": "approved",
            "workflow_stage": "approved"
        }
    else:
        return {
            "human_feedback": feedback,
            "workflow_stage": "needs_revision"
        }

def revision_node(state: HumanLoopState):
    """Revise content based on human feedback."""
    draft = state["pending_approval"]
    feedback = state["human_feedback"]
    
    revision_prompt = f"""
    Revise this draft based on the feedback:
    
    Original Draft:
    {draft}
    
    Feedback:
    {feedback}
    
    Provide an improved version.
    """
    
    response = llm.invoke([SystemMessage(content=revision_prompt)])
    
    return {
        "pending_approval": response.content,
        "workflow_stage": "awaiting_review",
        "iteration_count": state["iteration_count"] + 1
    }

def finalize_node(state: HumanLoopState):
    """Finalize approved content."""
    approved_content = state["pending_approval"]
    
    return {
        "messages": [AIMessage(content=f"Final approved content:\n\n{approved_content}")],
        "workflow_stage": "complete"
    }

def should_continue_review(state: HumanLoopState):
    """Determine next step based on review status."""
    stage = state["workflow_stage"]
    iterations = state["iteration_count"]
    
    if stage == "awaiting_review":
        return "human_review"
    elif stage == "needs_revision":
        if iterations < 3:  # Limit revisions
            return "revision"
        else:
            return "finalize"  # Force finalization after too many iterations
    elif stage == "approved":
        return "finalize"
    else:
        return "end"

# Create human-in-the-loop agent
def create_human_loop_agent():
    # Set up checkpointer for persistence
    checkpointer = MemorySaver()
    
    workflow = StateGraph(HumanLoopState)
    
    # Add nodes
    workflow.add_node("draft", draft_content_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("revision", revision_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("draft")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "draft",
        should_continue_review,
        {
            "human_review": "human_review"
        }
    )
    
    workflow.add_conditional_edges(
        "human_review",
        should_continue_review,
        {
            "revision": "revision",
            "finalize": "finalize"
        }
    )
    
    workflow.add_conditional_edges(
        "revision", 
        should_continue_review,
        {
            "human_review": "human_review",
            "finalize": "finalize"
        }
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile(checkpointer=checkpointer)

# Usage example (interactive)
def test_human_loop_agent():
    agent = create_human_loop_agent()
    
    # Configure for persistence
    config = {"configurable": {"thread_id": "human-loop-1"}}
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Write a blog post about LangGraph benefits")],
        "pending_approval": "",
        "human_feedback": "",
        "iteration_count": 0,
        "workflow_stage": "start"
    }, config=config)
    
    print("\n=== Final Result ===")
    print(result["messages"][-1].content)

# Uncomment to test interactively
# test_human_loop_agent()
```

---

</details>

---

## State Persistence and Memory

<details>
<summary>Advanced State Management and Checkpoint Configuration</summary>

---

### Configuring Persistence

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
import sqlite3

# Memory-based persistence (for development)
memory_checkpointer = MemorySaver()

# SQLite persistence (for production)
def create_sqlite_checkpointer(db_path="checkpoints.db"):
    # Create connection
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)

# Persistent state example
class PersistentAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_preferences: Dict[str, any]
    conversation_history: List[Dict]
    session_metadata: Dict[str, any]

def conversation_node(state: PersistentAgentState):
    """Handle conversation with persistent context."""
    messages = state["messages"]
    preferences = state.get("user_preferences", {})
    
    # Customize response based on stored preferences
    system_prompt = f"""
    You are a helpful assistant. 
    User preferences: {preferences}
    Consider these preferences in your response.
    """
    
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    
    # Update conversation history
    new_history_entry = {
        "timestamp": "2024-01-01T12:00:00",  # In production, use actual timestamp
        "user_message": messages[-1].content,
        "assistant_response": response.content
    }
    
    updated_history = state.get("conversation_history", []) + [new_history_entry]
    
    return {
        "messages": [response],
        "conversation_history": updated_history
    }

def preference_update_node(state: PersistentAgentState):
    """Update user preferences based on interaction."""
    messages = state["messages"]
    last_message = messages[-1].content.lower()
    
    current_prefs = state.get("user_preferences", {})
    
    # Simple preference detection (in production, use more sophisticated NLP)
    if "formal" in last_message:
        current_prefs["communication_style"] = "formal"
    elif "casual" in last_message:
        current_prefs["communication_style"] = "casual"
    
    if "technical" in last_message:
        current_prefs["detail_level"] = "technical"
    elif "simple" in last_message:
        current_prefs["detail_level"] = "simple"
    
    return {"user_preferences": current_prefs}

def create_persistent_agent():
    checkpointer = create_sqlite_checkpointer()
    
    workflow = StateGraph(PersistentAgentState)
    
    workflow.add_node("conversation", conversation_node)
    workflow.add_node("update_preferences", preference_update_node)
    
    workflow.set_entry_point("conversation")
    workflow.add_edge("conversation", "update_preferences")
    workflow.add_edge("update_preferences", END)
    
    return workflow.compile(checkpointer=checkpointer)

# Usage with persistent sessions
def test_persistent_agent():
    agent = create_persistent_agent()
    
    # Session 1
    session_1_config = {"configurable": {"thread_id": "user-123"}}
    
    result1 = agent.invoke({
        "messages": [HumanMessage(content="I prefer technical explanations. How does machine learning work?")],
        "user_preferences": {},
        "conversation_history": [],
        "session_metadata": {"user_id": "123"}
    }, config=session_1_config)
    
    print("=== Session 1 ===")
    print(f"Response: {result1['messages'][-1].content}")
    print(f"Preferences: {result1['user_preferences']}")
    
    # Session 2 (continuing same thread)
    result2 = agent.invoke({
        "messages": [HumanMessage(content="Tell me about neural networks")],
        "user_preferences": {},
        "conversation_history": [],
        "session_metadata": {}
    }, config=session_1_config)
    
    print("\n=== Session 2 (Same User) ===")
    print(f"Response: {result2['messages'][-1].content}")
    print(f"Stored Preferences: {result2['user_preferences']}")
    print(f"History Length: {len(result2['conversation_history'])}")

test_persistent_agent()
```

### Advanced State Patterns

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class WorkflowStage(Enum):
    INITIALIZATION = "initialization"
    PROCESSING = "processing"
    WAITING_FOR_INPUT = "waiting_for_input"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class TaskProgress:
    total_steps: int
    completed_steps: int
    current_step: str
    estimated_completion: Optional[str] = None
    
    @property
    def progress_percentage(self) -> float:
        return (self.completed_steps / self.total_steps) * 100

class AdvancedAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    workflow_stage: WorkflowStage
    task_progress: TaskProgress
    error_log: List[Dict[str, Any]]
    context_data: Dict[str, Any]
    sub_task_results: Dict[str, Any]
    user_session: Dict[str, Any]

def initialize_workflow_node(state: AdvancedAgentState):
    """Initialize complex workflow with multiple stages."""
    user_request = state["messages"][-1].content
    
    # Analyze request complexity
    analysis_prompt = f"""
    Analyze this request and estimate the work required:
    Request: {user_request}
    
    Provide:
    1. Number of steps needed (1-10)
    2. Estimated complexity (low/medium/high)
    3. Required sub-tasks
    
    Format as JSON.
    """
    
    response = llm.invoke([SystemMessage(content=analysis_prompt)])
    
    try:
        analysis = json.loads(response.content)
        total_steps = analysis.get("steps", 5)
    except:
        total_steps = 5
    
    progress = TaskProgress(
        total_steps=total_steps,
        completed_steps=0,
        current_step="initialization"
    )
    
    return {
        "workflow_stage": WorkflowStage.PROCESSING,
        "task_progress": progress,
        "context_data": {"user_request": user_request},
        "messages": [AIMessage(content=f"Starting workflow with {total_steps} estimated steps...")]
    }

def processing_node(state: AdvancedAgentState):
    """Handle main processing with progress tracking."""
    progress = state["task_progress"]
    context = state["context_data"]
    
    # Simulate processing step
    current_step = progress.completed_steps + 1
    step_name = f"step_{current_step}"
    
    # Update progress
    updated_progress = TaskProgress(
        total_steps=progress.total_steps,
        completed_steps=current_step,
        current_step=step_name
    )
    
    # Simulate some work
    step_result = f"Completed {step_name}: Processing data..."
    
    updated_context = context.copy()
    updated_context[f"result_{current_step}"] = step_result
    
    return {
        "task_progress": updated_progress,
        "context_data": updated_context,
        "messages": [AIMessage(content=f"Progress: {updated_progress.progress_percentage:.1f}% - {step_result}")]
    }

def should_continue_processing(state: AdvancedAgentState):
    """Determine if processing should continue."""
    progress = state["task_progress"]
    stage = state["workflow_stage"]
    
    if stage == WorkflowStage.PROCESSING:
        if progress.completed_steps < progress.total_steps:
            return "continue"
        else:
            return "finalize"
    elif stage == WorkflowStage.ERROR:
        return "error"
    else:
        return "end"

def finalization_node(state: AdvancedAgentState):
    """Finalize workflow and prepare results."""
    context = state["context_data"]
    progress = state["task_progress"]
    
    # Compile final results
    results_summary = []
    for i in range(1, progress.completed_steps + 1):
        result = context.get(f"result_{i}", f"Step {i} completed")
        results_summary.append(result)
    
    final_message = f"""
    Workflow completed successfully!
    
    Summary:
    - Total steps: {progress.total_steps}
    - Completed: {progress.completed_steps}
    - Progress: {progress.progress_percentage:.1f}%
    
    Results:
    {chr(10).join(f"- {result}" for result in results_summary)}
    """
    
    return {
        "workflow_stage": WorkflowStage.COMPLETED,
        "messages": [AIMessage(content=final_message)]
    }

def create_advanced_workflow():
    checkpointer = MemorySaver()
    
    workflow = StateGraph(AdvancedAgentState)
    
    workflow.add_node("initialize", initialize_workflow_node)
    workflow.add_node("process", processing_node)
    workflow.add_node("finalize", finalization_node)
    
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "process")
    
    workflow.add_conditional_edges(
        "process",
        should_continue_processing,
        {
            "continue": "process",
            "finalize": "finalize",
            "end": END
        }
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile(checkpointer=checkpointer)

# Test advanced workflow
def test_advanced_workflow():
    agent = create_advanced_workflow()
    
    config = {"configurable": {"thread_id": "advanced-workflow-1"}}
    
    # Initialize default state
    initial_progress = TaskProgress(total_steps=0, completed_steps=0, current_step="start")
    
    result = agent.invoke({
        "messages": [HumanMessage(content="Process this complex data analysis task")],
        "workflow_stage": WorkflowStage.INITIALIZATION,
        "task_progress": initial_progress,
        "error_log": [],
        "context_data": {},
        "sub_task_results": {},
        "user_session": {"user_id": "test_user"}
    }, config=config)
    
    print("=== Advanced Workflow Results ===")
    print(f"Final stage: {result['workflow_stage']}")
    print(f"Progress: {result['task_progress'].progress_percentage:.1f}%")
    print("\nFinal message:")
    print(result['messages'][-1].content)

test_advanced_workflow()
```

---

</details>

---

## Production Deployment and Monitoring

<details>
<summary>Enterprise-Grade Deployment Patterns and Observability</summary>

---

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import asyncio

# Pydantic models for API
class ChatRequest(BaseModel):
    message: str
    session_id: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    session_id: str
    stage: str
    metadata: Dict[str, Any]

class AgentAPI:
    def __init__(self):
        self.app = FastAPI(title="LangGraph Agent API", version="1.0.0")
        self.agent = create_persistent_agent()  # Your agent from previous examples
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/chat", response_model=ChatResponse)
        async def chat_endpoint(request: ChatRequest):
            try:
                config = {"configurable": {"thread_id": request.session_id}}
                
                result = await asyncio.create_task(
                    self.async_invoke(request, config)
                )
                
                return ChatResponse(
                    response=result["messages"][-1].content,
                    session_id=request.session_id,
                    stage=result.get("workflow_stage", "completed"),
                    metadata=result.get("session_metadata", {})
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "service": "langgraph-agent"}
        
        @self.app.get("/sessions/{session_id}/state")
        async def get_session_state(session_id: str):
            # Get current state for session
            config = {"configurable": {"thread_id": session_id}}
            try:
                # In production, you'd retrieve state from checkpointer
                return {"session_id": session_id, "status": "active"}
            except Exception as e:
                raise HTTPException(status_code=404, detail="Session not found")
    
    async def async_invoke(self, request: ChatRequest, config: Dict):
        """Async wrapper for agent invocation."""
        loop = asyncio.get_event_loop()
        
        state = {
            "messages": [HumanMessage(content=request.message)],
            "user_preferences": {},
            "conversation_history": [],
            "session_metadata": {
                "user_id": request.user_id,
                "session_id": request.session_id,
                **request.metadata
            }
        }
        
        # Run in thread pool to avoid blocking
        result = await loop.run_in_executor(
            None, 
            lambda: self.agent.invoke(state, config)
        )
        
        return result

# Production deployment setup
def create_production_app():
    agent_api = AgentAPI()
    return agent_api.app

app = create_production_app()

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  langgraph-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - DATABASE_URL=postgresql://user:password@postgres:5432/langgraph
    depends_on:
      - postgres
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=langgraph
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

### Monitoring and Observability

```python
import time
import logging
from typing import Any, Dict
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog

# Metrics
REQUEST_COUNT = Counter('langgraph_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('langgraph_request_duration_seconds', 'Request duration')
ACTIVE_SESSIONS = Gauge('langgraph_active_sessions', 'Number of active sessions')
WORKFLOW_STAGE = Counter('langgraph_workflow_stages_total', 'Workflow stages', ['stage'])

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class MonitoredAgent:
    def __init__(self, agent):
        self.agent = agent
        self.active_sessions = set()
    
    def invoke_with_monitoring(self, state: Dict[str, Any], config: Dict[str, Any]):
        """Invoke agent with comprehensive monitoring."""
        session_id = config.get("configurable", {}).get("thread_id")
        start_time = time.time()
        
        # Track active session
        if session_id:
            self.active_sessions.add(session_id)
            ACTIVE_SESSIONS.set(len(self.active_sessions))
        
        try:
            logger.info(
                "agent_request_started",
                session_id=session_id,
                message_count=len(state.get("messages", [])),
                state_keys=list(state.keys())
            )
            
            # Invoke agent
            with REQUEST_DURATION.time():
                result = self.agent.invoke(state, config)
            
            # Log success
            duration = time.time() - start_time
            REQUEST_COUNT.labels(method='invoke', endpoint='agent', status='success').inc()
            
            # Track workflow stage
            if "workflow_stage" in result:
                WORKFLOW_STAGE.labels(stage=str(result["workflow_stage"])).inc()
            
            logger.info(
                "agent_request_completed",
                session_id=session_id,
                duration=duration,
                response_message_count=len(result.get("messages", [])),
                workflow_stage=result.get("workflow_stage")
            )
            
            return result
            
        except Exception as e:
            REQUEST_COUNT.labels(method='invoke', endpoint='agent', status='error').inc()
            
            logger.error(
                "agent_request_failed",
                session_id=session_id,
                error=str(e),
                duration=time.time() - start_time,
                exc_info=True
            )
            raise
        
        finally:
            # Clean up session tracking
            if session_id and session_id in self.active_sessions:
                self.active_sessions.discard(session_id)
                ACTIVE_SESSIONS.set(len(self.active_sessions))

# Enhanced FastAPI with monitoring
class MonitoredAgentAPI(AgentAPI):
    def __init__(self):
        super().__init__()
        self.monitored_agent = MonitoredAgent(self.agent)
        
        # Start Prometheus metrics server
        start_http_server(8001)
        
        # Add monitoring middleware
        @self.app.middleware("http")
        async def monitor_requests(request, call_next):
            start_time = time.time()
            
            response = await call_next(request)
            
            duration = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration
            )
            
            return response
    
    async def async_invoke(self, request: ChatRequest, config: Dict):
        """Override with monitoring."""
        loop = asyncio.get_event_loop()
        
        state = {
            "messages": [HumanMessage(content=request.message)],
            "user_preferences": {},
            "conversation_history": [],
            "session_metadata": {
                "user_id": request.user_id,
                "session_id": request.session_id,
                **request.metadata
            }
        }
        
        # Use monitored agent
        result = await loop.run_in_executor(
            None,
            lambda: self.monitored_agent.invoke_with_monitoring(state, config)
        )
        
        return result

# Create production app with monitoring
def create_production_app_with_monitoring():
    return MonitoredAgentAPI().app

# Usage
if __name__ == "__main__":
    app = create_production_app_with_monitoring()
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

</details> 