import json
import logging
from typing import Dict, List
from dataclasses import dataclass
from pydantic import BaseModel

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
LITELLM_PROXY_URL = "http://localhost:4000"
DEFAULT_MODEL = "gpt-4o"

app = FastAPI(
    title="LangGraph Story Generator",
    description="A collaborative story generator using LangGraph and LiteLLM",
    version="1.0.0"
)

# Pydantic models for API
class StoryRequest(BaseModel):
    theme: str
    model: str = DEFAULT_MODEL

class StoryResponse(BaseModel):
    theme: str
    outline: str
    story: str
    final_story: str
    metadata: Dict

class StoryState(TypedDict):
    """State for the story generation workflow"""
    theme: str
    outline: str
    story: str
    reviewed_story: str
    current_step: str
    metadata: Dict

@dataclass
class LiteLLMClient:
    """Simple synchronous client for LiteLLM proxy"""
    base_url: str = LITELLM_PROXY_URL
    model: str = DEFAULT_MODEL
    api_key: str = "sk-1234"
    
    def chat(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """Send chat request to LiteLLM proxy (synchronous)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": 1000
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to connect to LiteLLM proxy at {self.base_url}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            if e.response.status_code == 401:
                raise HTTPException(status_code=401, detail="Authentication failed with LiteLLM proxy")
            raise HTTPException(status_code=502, detail=f"LiteLLM proxy error: {e.response.status_code}")

class StoryWorkflow:
    """Simplified LangGraph workflow for FastAPI"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.llm = LiteLLMClient(model=model)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the story generation workflow"""
        workflow = StateGraph(StoryState)
        
        workflow.add_node("create_outline", self.create_outline)
        workflow.add_node("write_story", self.write_story)
        workflow.add_node("review_story", self.review_story)
        
        workflow.add_edge(START, "create_outline")
        workflow.add_edge("create_outline", "write_story")
        workflow.add_edge("write_story", "review_story")
        workflow.add_edge("review_story", END)
        
        return workflow.compile()
    
    def create_outline(self, state: StoryState) -> StoryState:
        """Node 1: Create story outline"""
        logger.info("📝 Creating story outline...")
        
        messages = [
            {
                "role": "system", 
                "content": "You are a creative story planner. Create a brief but engaging story outline (3-4 sentences) based on the given theme. Include main characters, setting, and basic plot structure."
            },
            {
                "role": "user", 
                "content": f"Theme: {state['theme']}"
            }
        ]
        
        outline = self.llm.chat(messages, temperature=0.8)
        state["outline"] = outline
        state["current_step"] = "outline_created"
        
        return state
    
    def write_story(self, state: StoryState) -> StoryState:
        """Node 2: Write the actual story"""
        logger.info("✍️ Writing story...")
        
        messages = [
            {
                "role": "system",
                "content": "You are a skilled storyteller. Write a complete short story (300-500 words) based on the provided outline. Make it engaging with good dialogue and descriptive scenes."
            },
            {
                "role": "user",
                "content": f"Theme: {state['theme']}\n\nOutline: {state['outline']}\n\nWrite the story:"
            }
        ]
        
        story = self.llm.chat(messages, temperature=0.7)
        state["story"] = story
        state["current_step"] = "story_written"
        
        return state
    
    def review_story(self, state: StoryState) -> StoryState:
        """Node 3: Review and improve the story"""
        logger.info("🔍 Reviewing story...")
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional editor. Review the story and make improvements to enhance clarity, flow, and engagement. Fix any issues and polish the language. Return the improved version."
            },
            {
                "role": "user", 
                "content": f"Original story to review:\n\n{state['story']}"
            }
        ]
        
        reviewed_story = self.llm.chat(messages, temperature=0.3)
        state["reviewed_story"] = reviewed_story
        state["current_step"] = "completed"
        
        return state
    
    def generate_story(self, theme: str) -> Dict:
        """Generate a story synchronously"""
        logger.info(f"🚀 Starting story generation for theme: {theme}")
        
        initial_state = StoryState(
            theme=theme,
            outline="",
            story="",
            reviewed_story="",
            current_step="started",
            metadata={"model": self.llm.model}
        )
        
        try:
            result = self.graph.invoke(initial_state)
            logger.info("✅ Story generation completed!")
            
            return {
                "theme": result["theme"],
                "outline": result["outline"],
                "story": result["story"],
                "final_story": result["reviewed_story"],
                "metadata": result["metadata"]
            }
        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Story generation failed: {str(e)}")

# API Routes
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Simple HTML interface"""
    return """
    <html>
        <head>
            <title>LangGraph Story Generator</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                .form-group { margin: 15px 0; }
                input, button { padding: 10px; margin: 5px; }
                button { background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background-color: #0056b3; }
                .result { background-color: #f8f9fa; padding: 20px; border-radius: 4px; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>🎭 LangGraph Story Generator</h1>
            <p>Generate collaborative stories using LangGraph workflow</p>
            
            <div class="form-group">
                <input type="text" id="theme" placeholder="Enter story theme..." style="width: 300px;" />
                <button onclick="generateStory()">Generate Story</button>
            </div>
            
            <div id="result" class="result" style="display: none;">
                <h3>Generated Story</h3>
                <div id="story-content"></div>
            </div>
            
            <script>
                async function generateStory() {
                    const theme = document.getElementById('theme').value;
                    if (!theme) return alert('Please enter a theme');
                    
                    document.getElementById('result').style.display = 'none';
                    
                    try {
                        const response = await fetch('/generate-story', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ theme: theme })
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok) {
                            document.getElementById('story-content').innerHTML = `
                                <h4>Theme: ${result.theme}</h4>
                                <h5>Outline:</h5>
                                <p>${result.outline}</p>
                                <h5>Final Story:</h5>
                                <p style="white-space: pre-wrap;">${result.final_story}</p>
                            `;
                            document.getElementById('result').style.display = 'block';
                        } else {
                            alert('Error: ' + result.detail);
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                }
            </script>
        </body>
    </html>
    """

@app.post("/generate-story", response_model=StoryResponse)
def generate_story_endpoint(request: StoryRequest):
    """Generate a story based on theme"""
    workflow = StoryWorkflow(model=request.model)
    result = workflow.generate_story(request.theme)
    return StoryResponse(**result)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "langgraph-story-generator"}

@app.get("/example-themes")
def get_example_themes():
    """Get example story themes"""
    return {
        "themes": [
            "A robot discovers emotions",
            "Time traveler stuck in the wrong century",
            "Detective solving crimes with magic",
            "Friendship between human and alien",
            "Adventure in a world where books come alive"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 