# langchain-docs

## Agents      
Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use, and iteratively work towards solutions. 
  
Agents takes model, tools and system_prompt  

Lets check these component one by one.  

### Agent - Model  
The model is the reasoning engine of your agent.  
Its like a brain of AI agent which can read your question (it takes your input as text),, analyze it using patterns it has learned from data, decide which tool or action best   matches the request (based on those patterns), then respond with the predicted best answer.   

Models can be utilized in two ways:  
With agents - Models can be dynamically specified when creating an agent.  
Standalone - Models can be called directly (outside of the agent loop) for tasks like text generation, classification, or extraction without the need for an agent framework.  

With agents  
1. You can declare model name in create_agent using **model identifier string**. Its a string that follows the format `provider:model` (e.g. openai:gpt-5)
```
from langchain.agent import create_agent  
agent= create_agent(model="ollama:llama3.1")

result= agent.invoke(
  {"messages: ["role":"user", "content", "How is weather in Frankfurt?"]
)

print(f"result is {result['messages[[-1].content)
```

2. Using provider packages  
For more control over the model configuration, initialize a model instance directly using the provider package.
Following are some of the configuration parameters
```
from langchain_ollama import ChatOllama  # use model provider package

model = ChatOllama(
    model="llama3.1",
    temperature=0,      # deterministic, precise
    max_tokens=1000,    # allow up to ~750 words
    timeout=30,          # stop if it takes longer than 30s


```



Standalone   
Init_chat model    
The easiest way to get started with a standalone model in LangChain is to use init_chat_model to initialize one from a chat model provider of your choice.    
init_chat_model()  is the **recommended approach** for initializing models in LangChain.   
  
Why it's recommended:    
Unified interface - Works across different providers without changing code    
Flexibility - Easy to switch between providers (OpenAI, Anthropic, Azure, etc.)    
Simpler syntax - Less boilerplate than direct instantiation    

```
from langchain.chat_models import init_chat_model
model= init_chat_model("gpt-4.1")

response = model.invoke("Why do parrots talk?")
```  
Parameters    
 A chat model takes parameters that can be used to configure its behavior. The full set of supported parameters varies by model and provider, but standard ones include:  
  model: string required  
   The name or identifier of the specific model you want to use with a provider.  
    
  api_key: string
   The key required for authenticating with the model’s provider. This is usually issued when you sign up for access to the model. Often accessed by setting an environment variable.
     
  🧠 Temperature:    
    Purpose: Controls randomness / creativity of the model’s responses.    
    Temperature	Behavior         
    0	🔒 Deterministic — always produces the same output for the same input, 	Perfect for factual Q&A or coding tasks      
    0.7	⚖️ Balanced — a mix of consistency and creativity, 	Good for summarization or brainstorming      
    1.0+	🎨 Highly creative / random — more diverse wording, but sometimes less accurate	Useful for poetry, story generation
        
  🧮 max_tokens:    
    Purpose: Defines the maximum number of tokens the model can generate in its response.    
    Think of tokens as chunks of text — roughly 1 token ≈ 4 characters in English, or about ¾ of a word.    
    For example:    
    “ChatGPT is amazing!” → ~4 tokens.    
    1000 tokens ≈ 750 words.    
    1 token = 0.75 words   
    ✅ Tips:  
    max_tokens does not work for local model such ollama. For such local models there will be model specific parameter.    
    In case of ollama its num_predict
      
  ⏱️ timeout  
     Purpose: Sets a maximum time (in seconds) the model is allowed to take before throwing a timeout error.  
     Example: timeout=30 means if Ollama hasn’t responded in 30 seconds, it raises an error instead of hanging.  
     Useful when you’re dealing with slow local models (like large llama3 variants) or in web apps.   
      ✅ Tips:  
      Default is usually None (wait forever).  
      For interactive scripts, use 30–60 seconds.  
      For API servers, use something shorter (like 10–15s). 
        
  max_retries: Maximum number of retry attempts for failed requests.  
       When your code sends a request to a model API (like Ollama, OpenAI, etc.), sometimes the request fails — due to:  
           temporary network glitches 💻,  
           model server overload 🧠,  
           or timeout errors ⏱️.    
     Instead of crashing immediately, the LangChain client can automatically retry the request a few times before giving up.       
     That’s what max_retries controls — how many times it will retry.  
       
   base_url: Custom API endpoint URL. It tells LangChain where to send the API requests — i.e., the endpoint where your model is running and can be accessed.        
    
   rate_limiter: A BaseRateLimiter instance to control request rate.  
   its an an optional parameter that lets you control how often your code sends requests to an API (or local model like Ollama).  
   It accepts a BaseRateLimiter object — a built-in LangChain utility that keeps your requests under a certain rate (like “no more than 5 requests per second”).  
   This is critical when:  
     You’re hitting APIs that have usage limits (e.g., OpenAI, Anthropic, Hugging Face, etc.), or  
     You want to avoid overloading your local model or server.  
   
   Putting it all togther  
```
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_ollama import ChatOllama  # use model provider package

# Allow max 5 requests per second
rate_limiter = InMemoryRateLimiter(requests_per_second=5)

model = ChatOllama(
    model="llama3.1",
    temperature=0,      # deterministic, precise
    max_tokens=1000,    # allow up to ~750 words
    timeout=30,          # stop if it takes longer than 30s
    base_url="http://localhost:11434",  # Default Ollama endpoint
    max_retries=3 , # <-- Will retry up to 3 times on failure
    rate_limiter=rate_limiter

)
```  
 
   
#### Types of model    
  
##### Static model  
Static models are configured once when creating the agent and remain unchanged throughout execution. 
```
from langchain.agents import create_agent

agent = create_agent(
    "gpt-5",
    tools=tools
)
```
  
##### Dynamic model  
Dynamic models are selected at runtime based on the current state and context. **This enables sophisticated routing logic and cost optimization**

Runtime refers to the execution environment of your agent, containing immutable configuration and contextual data that persists throughout the agent's execution (e.g., user IDs,   session details, or application-specific configuration).  
  
 State refers to the data that flows through your agent's execution, including messages, custom fields, and any information that needs to be tracked and potentially modified   during processing (e.g., user preferences or tool usage stats).   
     
Context here refers to information the agent or model has available at the moment that can affect how it behaves or which model/tool it chooses.  

In simpler terms:

Context = all the relevant info the agent “knows” right now.  
Examples:  

The previous conversation in a chat.  

The user query you just sent.  

Data retrieved from a database or documents.  

The current situation/state of the workflow.  

To use a dynamic model, create middleware using the @wrap_model_call decorator that modifies the model in the request:  
```
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)

```  


### Agent - Tools
  
Tools give agents the ability to take actions. Agents go beyond simple model-only tool binding by facilitating:  
Multiple tool calls in sequence (triggered by a single prompt)  
Parallel tool calls when appropriate  
Dynamic tool selection based on previous results  
Tool retry logic and error handling  
State persistence across tool calls   

Many AI applications interact with users via natural language. However, some use cases require models to interface directly with external systems—such as APIs, databases, or file systems—using structured input.
Tools are components that agents call to perform actions. They extend model capabilities by letting them interact with the world through well-defined inputs and outputs.
  
Defining tools    
Tools can be specified as plain Python functions or coroutines.  
  
Tools are of 2 types    
1. server side tools  
2. client side tools


**server side tools**    
Server-side tools are provider-hosted tools that the model calls and executes internally in one turn — you get results without   executing anything yourself. Yes, they're only from providers, but agents can absolutely use them.    

The model provider (OpenAI, Anthropic, etc.) hosts certain tools that the model can invoke directly. The model runs the tool, gets   
the result, and analyzes it — all in a single API call. You never see or execute the tool yourself.  
```
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1-mini")

# Provider-hosted tool (OpenAI's web search)
model_with_tools = model.bind_tools([{"type": "web_search"}])

# ONE call: model searches internally, returns answer with results
response = model_with_tools.invoke("What was a positive news story today?")

# You get the final answer already analyzed
print(response.content_blocks)
# [
#     {"type": "server_tool_call", "name": "web_search", ...},
#     {"type": "server_tool_result", "status": "success", ...},
#     {"type": "text", "text": "Here are positive stories..."}
# ]
```
Who Provides Server-Side Tools?  
Only the AI model providers such as Openai,Anthropic, google etc.  

Can Agents Use Server-Side Tools?  
Yes, absolutely. Agents work seamlessly with server-side tools:  

```
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")

# Mix server-side and client-side tools
agent = create_agent(
    model=model,
    tools=[
        {"type": "web_search"},  # ✅ Server-side (provider-hosted)
        my_custom_database_tool,  # ✅ Client-side (you execute it)
        my_api_wrapper_tool,      # ✅ Client-side
    ],
    system_prompt="Research topics and fetch from our database"
)

# Agent automatically uses server-side when it needs web search
# Agent uses client-side for database queries
result = agent.invoke({
    "messages": [{"role": "user", "content": "Search the web for X and check our database"}]
})
```
Why You'd Use Each  
Use Server-Side Tools When:  
  
You need web search or code execution  
You trust the provider's implementation  
You want simplicity (one API call)  
  
Use Client-Side Tools When:  
  
You need custom logic (database queries, APIs)  
You want full control over execution  
You're integrating proprietary systems  
 

```
Full Comparison
Aspect	                    Server-Side Tools	                     Client-Side Tools
Import needed?	              ❌ No                     	           ✅ Yes
Definition	                {"type": "web_search"}      	           @tool def my_tool(): ...
Who runs it?	              Provider's servers	                      Your code
Available tools           	Limited (provider decides)	               Unlimited (you build them)
Customization             	❌ Not possible                       	  ✅ Full control
Example	{                   "type": "web_search"}	                     from langchain.tools import tool


```
Create tools  
The simplest way to create a tool is with the @tool decorator. By default, the function’s docstring becomes the tool’s description   
that helps the model understand when to use it.

A docstring is a text string that describes what a function does — it's placed right after the function definition and the model   
reads it to decide when to use the tool.  

```
Quick Checklist for Good Docstrings

@tool
def my_tool(param1: str, param2: int) -> str:
    """✅ Start with one-line summary explaining what the tool does.
    
    ✅ Add more details about when/why to use it (optional).
    
    ✅ Document each parameter with Args section.
    
    ✅ Explain what gets returned with Returns section.
    """
    return "result"

```
```
Inside a Real Agent

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def check_inventory(product_id: str) -> str:
    """Check product availability in our inventory system.
    
    Returns stock levels and warehouse locations for a given product.
    Use this when customers ask if something is in stock.
    
    Args:
        product_id: SKU or product ID (e.g., "PROD-12345")
        
    Returns:
        Stock level and warehouse info
    """
    return "In stock: 50 units in Warehouse A"

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[check_inventory],
    system_prompt="You are a helpful customer service agent"
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Is the laptop in stock?"}]
})

# Agent reads docstring, understands what check_inventory does,
# calls it, and returns: "Yes, we have 50 units in Warehouse A"

```  

Tool also require Type hints are required as they define the tool’s input schema. Type hints means type of arguments passed to tool and return type of   tool.  

```
from langchain.tools import tool

# ✅ WITH type hints (REQUIRED for tools)
@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the database."""
    return f"Found {limit} results for '{query}'"

# ❌ WITHOUT type hints (WILL NOT WORK as a tool)
@tool
def search_database(query, limit=10):
    """Search the database."""
    return f"Found {limit} results for '{query}'"

```
use **@wrap_tool_call** decorator to handle exception/error in tools  
```
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

model = init_chat_model(model="llama3.1", model_provider="ollama", num_predict=100, temperature=0)

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    if not city or city.strip() == "":
        raise ValueError("City name cannot be empty") # creates and throws an error with a custom message. It stops the function and tells Python "something went wrong".
# raise -  # Command: "Stop here and throw an error"
#ValueError  # Type of error for invalid values
#Common error types:

#ValueError — Wrong VALUE (e.g., invalid city name)
#TypeError — Wrong TYPE (e.g., got string, expected number)
#ZeroDivisionError — Divided by zero
#KeyError — Key doesn't exist in dictionary
#IndexError — Index out of range
#Exception — Generic catch-all error
    
    if not city.replace(" ", "").isalpha():
        raise ValueError(f"Invalid city name '{city}'. Only letters allowed")
    
    return f"Weather of {city} is sunny"

@wrap_tool_call  # ← Decorator activates here
def handle_tool_errors(request, handler):
    """Handle tool errors with custom messages"""
  
    # request — Information about the tool call (what tool, what arguments, etc.)
    # handler — A function that actually executes the tool
    # request contains:
    #   - request.tool_call["id"] → unique ID of this tool call
    #   - request.tool_call["name"] → name of the tool ("get_weather")
    #   - request.tool_call["args"] → arguments passed to tool ({"city": "dd123"})
    
    try:
        # ✅ Normal case: tool succeeds
        result = handler(request)  # Calls get_weather(city="dd123")
        print(f"✅ Tool succeeded: {result}")
        return result
    
    except Exception as e:
        # ❌ Error case: tool raises exception
        print(f"❌ Tool failed: {str(e)}")
        
        # Return error message to agent instead of crashing
        return ToolMessage(
            content=f"Please check your input and try again ({str(e)})",
            tool_call_id=request.tool_call["id"]  # Links error to original tool call, What happens:

#Agent knows exactly which tool call this error belongs to
#Agent can correlate error with the original tool invocation
#Agent can continue reasoning properly

# without tool_call_id
#Agent gets the error message but doesn't know WHICH tool call failed
#Agent might get confused about execution flow
#Can lead to incorrect reasoning or loops
        )

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[handle_tool_errors]
)

# TEST: Invalid input
print("=" * 60)
print("CALLING AGENT WITH INVALID INPUT")
print("=" * 60)
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What's weather in dd123?"
    }]
})
print("Agent response:", response["messages"][-1].content)

```
You can have multiple tools below @tool_call_wrap and cover exceptions/erros in all those tools as follows  
```
@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle errors differently based on which tool is being called"""
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})

  print("write a code here that you want to execute before tool call")
    
    try:
         result = handler(request)
         print("write a code here that you want to execute before tool call")
        return result
    
    except Exception as e:
        # ✅ Custom handling per tool
        if tool_name == "get_weather":
            return ToolMessage(
                content=f"Weather error: {str(e)}. Try a real city name.",
                tool_call_id=request.tool_call["id"]
            )
        
        elif tool_name == "convert_temperature":
            return ToolMessage(
                content=f"Conversion error: {str(e)}. Use 'fahrenheit' or 'kelvin'.",
                tool_call_id=request.tool_call["id"]
            )
        
        else:
            # Default for any other tools
            return ToolMessage(
                content=f"Error in {tool_name}: {str(e)}",
                tool_call_id=request.tool_call["id"]
            )

agent = create_agent(
    model=model,
    tools=[get_weather, get_temperature, convert_temperature],
    middleware=[handle_tool_errors]
)

```


```
Flow of tool execution 
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Agent: "I need to call get_weather(city='dd123')"    │
│                                                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ↓
        ┌─────────────────────────────┐
        │  @wrap_tool_call INTERCEPTS │
        │  (Middleware activates)     │
        └─────────────┬───────────────┘
                      │
                      ↓
            ┌─────────────────────┐
            │  try:               │
            │    handler(request) │ ← Executes the tool
            │                     │
            └────────────┬────────┘
                         │
          ┌──────────────┴──────────────┐
          ↓                             ↓
    ✅ SUCCEEDS                  ❌ FAILS (Exception)
    return result                    ↓
          │                    except Exception as e:
          │                    return ToolMessage(
          │                        content=error,
          │                        tool_call_id=id
          │                    )
          │                         │
          └────────────┬────────────┘
                       ↓
            ┌──────────────────────┐
            │ Agent receives result│
            │ or error message     │
            └──────────┬───────────┘
                       ↓
            ┌──────────────────────┐
            │ Agent responds       │
            │ to user              │
            └──────────────────────┘






┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: USER INVOKES AGENT                                      │
│ agent.invoke({"messages": [{"role": "user", "content": "..."}]})│
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────┐
        │ STEP 2: AGENT STARTS PROCESSING  │
        │ - Reads user message             │
        │ - Processes through LLM          │
        └──────────────┬───────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────┐
        │ STEP 3: AGENT DECIDES TO CALL TOOL   │
        │ Agent thinks: "I need to call tool X"│
        │ Gathers:                             │
        │ - Tool name: "get_weather"           │
        │ - Tool args: {"city": "NYC"}         │
        │ - Tool call ID: "call_abc123"        │
        └──────────────┬──────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 4: LANGCHAIN CREATES REQUEST OBJECT │
        │                                          │
        │ request = {                              │
        │   tool_call: {                           │
        │     "id": "call_abc123",                 │
        │     "name": "get_weather",               │
        │     "args": {"city": "NYC"}              │
        │   },                                     │
        │   runtime: {                             │
        │     state: {...},                        │
        │     context: {...},                      │
        │     store: {...}                         │
        │   }                                      │
        │ }                                        │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 5: @wrap_tool_call INTERCEPTS       │
        │ Middleware activates!                    │
        │ Calls: your_middleware(request, handler) │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 6: YOUR MIDDLEWARE RUNS (try block) │
        │                                          │
        │ @wrap_tool_call                          │
        │ def my_middleware(request, handler):     │
        │     try:                                 │
        │         # ← YOU ARE HERE                 │
        │         # Can inspect request:           │
        │         # - request.tool_call["name"]   │
        │         # - request.tool_call["args"]   │
        │         # - request.runtime.state       │
        │         # - request.runtime.context     │
        │                                          │
        │         result = handler(request)        │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 7: CALL handler(request)            │
        │ This is the gate to actual tool          │
        │ handler = function that executes tool    │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 8: TOOL EXECUTES                    │
        │ @tool                                    │
        │ def get_weather(city: str) -> str:       │
        │     # Tool function runs here            │
        │     if not city:                         │
        │         raise ValueError("Empty city")   │
        │     return f"Weather in {city}: Sunny"   │
        └──────────────┬──────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
    ✅ SUCCESS              ❌ EXCEPTION RAISED
    Tool returns result     Tool raises error
          │                         │
          ↓                         ↓
    "Weather in NYC:        ValueError:
     Sunny"                 "Empty city"
          │                         │
          ↓                         ↓
    ┌─────────────────┐    ┌──────────────────┐
    │ STEP 9a:        │    │ STEP 9b:         │
    │ SUCCESS PATH    │    │ ERROR PATH       │
    │                 │    │                  │
    │ result = "..."  │    │ Exception caught │
    │                 │    │ except Exception │
    │ Return result   │    │ as e:            │
    │ (auto-wrapped   │    │                  │
    │  in ToolMessage)│    │ Create error     │
    │                 │    │ ToolMessage      │
    └────────┬────────┘    └────────┬─────────┘
             │                      │
             ↓                      ↓
    ┌────────────────────────────────────────┐
    │ STEP 10: RETURN TO AGENT               │
    │ Middleware returns to agent:           │
    │                                        │
    │ ✅ Success case:                       │
    │ ToolMessage(content="Weather....",     │
    │            tool_call_id="call_...")    │
    │                                        │
    │ ❌ Error case:                         │
    │ ToolMessage(content="Error: ...",      │
    │            tool_call_id="call_...")    │
    └────────┬─────────────────────────────┘
             │
             ↓
    ┌────────────────────────────────────┐
    │ STEP 11: AGENT RECEIVES RESULT     │
    │ Agent sees:                        │
    │ - Tool execution succeeded         │
    │ OR                                 │
    │ - Tool execution failed (error msg)│
    │                                    │
    │ Agent adds to message history:     │
    │ - ToolMessage with result/error    │
    └────────┬─────────────────────────┘
             │
             ↓
    ┌────────────────────────────────────┐
    │ STEP 12: AGENT DECIDES NEXT STEP   │
    │ Agent thinks:                      │
    │ "Did tool succeed?"                │
    │                                    │
    │ If success:                        │
    │ - Use result in reasoning          │
    │ - May call more tools              │
    │ - Or generate final answer         │
    │                                    │
    │ If error:                          │
    │ - Read error message               │
    │ - Retry tool OR                    │
    │ - Try different approach           │
    └────────┬─────────────────────────┘
             │
             ↓
    ┌────────────────────────────────────┐
    │ STEP 13: FINAL RESPONSE TO USER    │
    │ Agent responds with final answer   │
    └────────────────────────────────────┘

```

Agent follows ReAct pattern. ReAct (Reasoning + Acting) is a pattern where the agent thinks through a problem step-by-step, then takes actions (calls tools), then reasons again   based on the results.  
  

What is ReAct?  
ReAct = "Reason" + "Act"  
  
The agent:  
  
Thinks (Reason): "What do I need to do?"  
Does (Act): Calls a tool to get information  
Thinks (Reason): "What does this result mean?"  
Repeats until it has the answer    

```
User: "What's the population of France's capital?"

Agent Reasoning:
  "The user is asking about France's capital.
   I need to find out:
   1. What is France's capital?
   2. What is its population?"

Agent Acting:
  Call tool: search("France capital")
  
Tool Result:
  "Paris is the capital of France"

Agent Reasoning:
  "Good! Now I know the capital is Paris.
   Now I need to find the population."

Agent Acting:
  Call tool: search("Paris population")
  
Tool Result:
  "Paris population is approximately 2.1 million"

Agent Reasoning:
  "Perfect! I now have both pieces of information.
   I can answer the user's question."

Agent Final Answer:
  "The capital of France is Paris, 
   and its population is approximately 2.1 million people."

```  


```
In Summary

ReAct Pattern = Thinking out loud while solving problems

Think → Act → Observe → Think → Act → Observe → ... → Answer
Just like how YOU solve a complex problem:

Think: "What information do I need?"
Act: "Let me search for it"
Observe: "Here's what I found"
Think: "What does this mean? Do I need more?"
Repeat or Answer


```

 **System prompt** tells the agent HOW to perform a task — it's like giving instructions/guidelines to the agent.  
 What System Prompt Does  
 system_prompt = "You are a helpful weather assistant. Always call the get_weather tool when asked about weather."    
 This tells the agent:  
  
WHO you are (helpful weather assistant)  
WHAT to do (call get_weather tool)  
HOW to behave (helpfully)    

```
Sytem Prompt Vs user message

# SYSTEM PROMPT: Instructions for the agent
system_prompt = "You are a weather assistant. Be concise and accurate."

# USER MESSAGE: What the user is asking
user_message = "What's the weather in New York?"

# Agent thinks:
# "My system prompt says I'm a weather assistant.
#  The user is asking about weather.
#  I should call my weather tool."

```  
 When no system_prompt is provided, the agent will infer its task from the messages directly.  Output: May be unpredictable, agent might not use the tool.  

 ```
# ✅ BEST PRACTICE: Always include tool guidance in system prompt
agent = create_agent(
    model=model,
    tools=[tool1, tool2, tool3],
    system_prompt="""You are [role].

Tool usage:
- Use [tool1] when [condition]
- Use [tool2] when [condition]
- Use [tool3] when [condition]

Always prioritize [some tool] for accuracy."""
)

# Why? Because:
# 1. Ensures consistent tool usage
# 2. Prevents agent hallucination (making up answers)
# 3. Better control over agent behavior
# 4. Clearer expectations
# 5. Easier to debug issues


```
  
Dynamic system prompt  
Dynamic system prompt means the system prompt CHANGES based on runtime conditions — it's not fixed, it adapts to the situation.  
  
Instead of one static prompt, the prompt is generated on-the-fly based on:  
  
User data  
Agent state  
Context  
Current conditions    

For more advanced use cases where you need to modify the system prompt based on runtime context or agent state, you can use middleware.    
The @dynamic_prompt decorator creates middleware that generates system prompts based on the model request  


```
How Dynamic Prompt Works

User provides context
        ↓
Agent starts
        ↓
@dynamic_prompt runs
        ↓
Function generates prompt based on:
├─ User role
├─ Conversation history
├─ Message content
├─ Time
├─ User tier/status
└─ Any other condition
        ↓
Generated prompt is used
        ↓
Agent behaves according to dynamic prompt


Request Flow in Dynamic Prompt


User invokes agent
        ↓
@dynamic_prompt INTERCEPTS (before model call)
        ↓
request created with:
├── request.messages (conversation history)
├── request.state (agent state)
└── request.runtime (context, store)
        ↓
your_dynamic_prompt(request)
        ↓
READ request data to generate prompt
        ↓
RETURN string (system prompt)
        ↓
System prompt sent to model
        ↓
Model processes with dynamic prompt
        ↓
Model generates response




┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: USER INVOKES AGENT                                      │
│ agent.invoke(                                                   │
│     {"messages": [{"role": "user", "content": "Explain AI"}]},  │
│     context=UserContext(user_level="beginner")                  │
│ )                                                               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────┐
        │ STEP 2: AGENT STARTS PROCESSING  │
        │ - Receives user input            │
        │ - Reads provided context         │
        │ - Prepares to call LLM           │
        └──────────────┬───────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 3: ABOUT TO CALL MODEL              │
        │ Agent has:                               │
        │ - User messages: [HumanMessage(...)]     │
        │ - User context: UserContext(...)         │
        │ - Agent state: {...}                     │
        │                                          │
        │ Thinks: "I need to call the LLM now"     │
        │ But BEFORE calling, middleware intercepts│
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 4: LANGCHAIN CREATES REQUEST OBJECT │
        │                                          │
        │ request = {                              │
        │   messages: [                            │
        │     HumanMessage(...),                   │
        │     AIMessage(...),                      │
        │     ...                                  │
        │   ],                                     │
        │   state: {                               │
        │     "messages": [...],                   │
        │     "custom_field": "value"              │
        │   },                                     │
        │   runtime: {                             │
        │     context: UserContext(...),           │
        │     store: InMemoryStore(...),           │
        │     state: {...}                         │
        │   }                                      │
        │ }                                        │
        │                                          │
        │ This request contains ALL info about:    │
        │ - What will be sent to model             │
        │ - Who is sending it (context)            │
        │ - What state they're in                  │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 5: @dynamic_prompt INTERCEPTS       │
        │ Middleware activates!                    │
        │ Calls: your_middleware(request)          │
        │                                          │
        │ NOTE: NO handler parameter!              │
        │ NO try/except needed                     │
        │ Just return a string (the prompt)        │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 6: YOUR MIDDLEWARE RUNS             │
        │                                          │
        │ @dynamic_prompt                          │
        │ def my_middleware(request):              │
        │     # ← YOU ARE HERE                     │
        │     # Can inspect request:               │
        │     # - request.messages                 │
        │     # - request.state                    │
        │     # - request.runtime.context          │
        │                                          │
        │     # DECIDE what prompt to generate     │
        │     if condition:                        │
        │         return "Expert prompt"           │
        │     else:                                │
        │         return "Beginner prompt"         │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 7: INSPECT REQUEST DATA             │
        │                                          │
        │ Get messages from request:               │
        │ messages = request.messages              │
        │ └─ List of Message objects               │
        │    - HumanMessage                        │
        │    - AIMessage                           │
        │    - ToolMessage                         │
        │    - etc.                                │
        │                                          │
        │ Get state from request:                  │
        │ state = request.state                    │
        │ └─ Current agent state dict              │
        │    - "messages": [...]                   │
        │    - "topic": "Python"                   │
        │    - "user_level": "beginner"            │
        │                                          │
        │ Get context from request:                │
        │ context = request.runtime.context        │
        │ └─ User-provided context object          │
        │    - user_level = "beginner"             │
        │    - user_name = "Alice"                 │
        │    - loyalty_tier = "gold"               │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 8: MAKE DECISIONS BASED ON DATA     │
        │                                          │
        │ Check 1: Who is the user?                │
        │ if context.user_level == "expert":       │
        │     → Use expert prompt                  │
        │ else:                                    │
        │     → Use beginner prompt                │
        │                                          │
        │ Check 2: How long is conversation?       │
        │ if len(messages) > 10:                   │
        │     → Use concise prompt                 │
        │ else:                                    │
        │     → Use detailed prompt                │
        │                                          │
        │ Check 3: What are they asking?           │
        │ if "urgent" in messages[-1].content:     │
        │     → Use speed prompt                   │
        │ elif "explain" in messages[-1].content:  │
        │     → Use educational prompt             │
        │                                          │
        │ Combine all checks...                    │
        │ Generated Prompt = "..."                 │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 9: RETURN GENERATED PROMPT          │
        │                                          │
        │ return """You are an expert assistant.   │
        │ Provide technical responses with         │
        │ advanced terminology and deep analysis."""│
        │                                          │
        │ Returned as: STRING (not object)         │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 10: LANGCHAIN PREPARES MODEL INPUT  │
        │                                          │
        │ Combines:                                │
        │ 1. Generated system_prompt (from above)  │
        │ 2. User messages (from request)          │
        │                                          │
        │ Final input to model:                    │
        │ [                                        │
        │   SystemMessage(content="Expert..."),    │
        │   HumanMessage(content="Explain AI"),    │
        │   ...                                    │
        │ ]                                        │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 11: MODEL RECEIVES INPUT            │
        │                                          │
        │ Model now knows:                         │
        │ "I'm an expert assistant."               │
        │ "Answer with technical depth."           │
        │ "Provide advanced analysis."             │
        │                                          │
        │ User asked: "Explain AI"                 │
        │                                          │
        │ Model generates response considering:    │
        │ - The dynamic system prompt              │
        │ - The user's expertise level             │
        │ - Previous conversation history          │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 12: MODEL PROCESSES & RESPONDS      │
        │                                          │
        │ Model output (tailored to prompt):       │
        │ "Artificial Intelligence represents...   │
        │  In machine learning, neural networks    │
        │  utilize backpropagation algorithms...   │
        │  [Advanced technical explanation]"       │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 13: RESPONSE ADDED TO HISTORY       │
        │                                          │
        │ Update messages:                         │
        │ messages.append(                         │
        │   AIMessage(content="AI represents...")  │
        │ )                                        │
        │                                          │
        │ New state:                               │
        │ [                                        │
        │   HumanMessage("Explain AI"),            │
        │   AIMessage("AI represents... [expert]") │
        │ ]                                        │
        └──────────────┬──────────────────────────┘
                       │
                       ↓
        ┌──────────────────────────────────────────┐
        │ STEP 14: FINAL RESPONSE TO USER          │
        │                                          │
        │ User receives expert-level explanation   │
        │ because dynamic prompt was generated     │
        │ based on their context (user_level=...) │
        └──────────────────────────────────────────┘

```


```
# Function that GENERATES the prompt dynamically
@dynamic_prompt
def personalized_prompt(request):
    """Generate prompt based on user level"""
    user_level = request.runtime.context.get("user_level", "beginner")
    
    if user_level == "expert":
        return "You are an expert meteorologist. Provide detailed technical analysis."
    else:
        return "You are a helpful weather assistant. Keep it simple."

agent = create_agent(
    model=model,
    tools=[get_weather],
    middleware=[personalized_prompt],
    context_schema={"user_level": str}
)

# Expert user gets expert prompt
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Weather?"}]},
    context={"user_level": "expert"}  # ← Different context
)

# Beginner user gets beginner prompt
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Weather?"}]},
    context={"user_level": "beginner"}  # ← Different context
)


```


```
Dynamic System Prompt Based on User Role

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from dataclasses import dataclass

model = init_chat_model(model="llama3.1", model_provider="ollama", num_predict=100, temperature=0)

@tool
def access_database(query: str) -> str:
    """Access company database"""
    return f"Database result for: {query}"

@tool
def view_salary(employee_id: str) -> str:
    """View employee salary"""
    return f"Salary: $50,000"

@tool
def fire_employee(employee_id: str) -> str:
    """Fire an employee"""
    return f"Employee {employee_id} removed"

# Define context schema
@dataclass
class UserContext:
    user_role: str  # "admin", "manager", "employee"
    user_name: str

# ✅ DYNAMIC PROMPT: Changes based on user role
@dynamic_prompt
def role_based_prompt(request) -> str:
    """Generate prompt based on user role"""
    context = request.runtime.context
    user_role = context.user_role
    user_name = context.user_name
    
    if user_role == "admin":
        return f"""You are an Admin Assistant for {user_name}.
        
You have full access to all tools:
- access_database: Query any data
- view_salary: Check employee salaries
- fire_employee: Remove employees

You can do anything. Be thorough."""
    
    elif user_role == "manager":
        return f"""You are a Manager Assistant for {user_name}.
        
You have limited access:
- access_database: Query company data
- view_salary: Check your team's salaries

You CANNOT:
- fire_employee (requires admin)

Be professional and follow company policy."""
    
    else:  # employee
        return f"""You are an Employee Assistant for {user_name}.
        
You have minimal access:
- access_database: View your own data only

You CANNOT:
- view_salary (confidential)
- fire_employee (not authorized)

Protect confidentiality."""

agent = create_agent(
    model=model,
    tools=[access_database, view_salary, fire_employee],
    middleware=[role_based_prompt],
    context_schema=UserContext
)

print("=" * 70)
print("ADMIN USER")
print("=" * 70)
response_admin = agent.invoke(
    {"messages": [{"role": "user", "content": "What can I do?"}]},
    context=UserContext(user_role="admin", user_name="Alice")
)
print(response_admin["messages"][-1].content)

print("\n" + "=" * 70)
print("MANAGER USER")
print("=" * 70)
response_manager = agent.invoke(
    {"messages": [{"role": "user", "content": "What can I do?"}]},
    context=UserContext(user_role="manager", user_name="Bob")
)
print(response_manager["messages"][-1].content)

print("\n" + "=" * 70)
print("EMPLOYEE USER")
print("=" * 70)
response_employee = agent.invoke(
    {"messages": [{"role": "user", "content": "What can I do?"}]},
    context=UserContext(user_role="employee", user_name="Charlie")
)
print(response_employee["messages"][-1].content)
```

```
Dynamic Prompt Based on Conversation History

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def history_based_prompt(request):
    """Prompt changes based on conversation length"""
    messages = request.state["messages"]
    message_count = len(messages)
    
    if message_count == 1:
        # First message - be extra friendly
        return "You are a friendly assistant. Welcome the user warmly!"
    
    elif message_count < 5:
        # Early conversation - help guide
        return "You are a helpful guide. Help the user get started."
    
    elif message_count < 15:
        # Mid conversation - be more direct
        return "You are a professional assistant. Be direct and efficient."
    
    else:
        # Long conversation - assume expertise
        return "You are an expert. Assume the user knows basics. Be advanced."

agent = create_agent(
    model=model,
    tools=[my_tool],
    middleware=[history_based_prompt]
)

# First message - friendly tone
response1 = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})

# Later messages - more advanced tone
messages = response1["messages"]
messages.append({"role": "user", "content": "Next question"})
response2 = agent.invoke({"messages": messages})
# Prompt automatically changes to more advanced
```

 ```
Dynamic Prompt Based on Message Content

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

@dynamic_prompt
def content_based_prompt(request):
    """Prompt changes based on what user is asking"""
    latest_message = request.state["messages"][-1]
    content = latest_message.get("content", "").lower()
    
    if "urgent" in content or "emergency" in content:
        return "URGENT MODE: Prioritize this request. Act fast. Be brief."
    
    elif "explain" in content or "teach" in content:
        return "EDUCATION MODE: Be thorough. Explain concepts. Use examples."
    
    elif "quick" in content or "fast" in content:
        return "SPEED MODE: Give the shortest accurate answer. No fluff."
    
    else:
        return "NORMAL MODE: Be balanced. Helpful but concise."

agent = create_agent(
    model=model,
    tools=[my_tool],
    middleware=[content_based_prompt]
)

# Urgent request - fast response
agent.invoke({"messages": [{"role": "user", "content": "URGENT: Fix this now!"}]})

# Learning request - detailed response
agent.invoke({"messages": [{"role": "user", "content": "Explain how this works"}]})

# Quick request - brief response
agent.invoke({"messages": [{"role": "user", "content": "Quick answer?"}]})
```

```
Dynamic Prompt Based on Time

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from datetime import datetime

@dynamic_prompt
def time_based_prompt(request):
    """Prompt changes based on time of day"""
    hour = datetime.now().hour
    
    if 6 <= hour < 12:
        return "Good morning! You are energetic and ready to help. Be enthusiastic!"
    
    elif 12 <= hour < 17:
        return "It's afternoon. You are focused and professional. Be efficient."
    
    elif 17 <= hour < 21:
        return "It's evening. You are helpful but winding down. Be friendly."
    
    else:
        return "It's late. You are calm and measured. Keep responses concise."

agent = create_agent(
    model=model,
    tools=[my_tool],
    middleware=[time_based_prompt]
)

# Morning - enthusiastic
agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})

# Night - calm
agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})

```







  



#### Invocation   
A chat model must be invoked to generate an output. There are three primary invocation methods, each suited to different use cases.  
Invoke, Stream, Batch  
  
###### Invoke  
The most straightforward way to call a model is to use invoke() with a single message or a list of messages.

```
from langchain.chat_models import init_chat_model

llm = init_chat_model(model="llama3.1", model_provider="ollama", num_predict=30, temperature=0)

response = llm.invoke("who is president of usa?")

print(f"president is {response}")

output: reponse is an AI message and has following format
{
    "content": "...", # contains AI answer
    "additional_kwargs": {},  # arguments/call to tool
    "response_metadata": {...}, # cotains info about response such as created_at, total_duration, prompt_eval_duration
    "id": "...",
    "usage_metadata": {...} # contains models token info such as input_tokens, output_tokens, total_tokens
}
```
A list of messages can be provided to a model to represent conversation history. Each message has a role that models use to indicate   who sent the message in the conversation.  
2 ways to passing list of messages to model  
```
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")
```

```
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")


```

###### Stream

Most models can stream their output content while it is being generated. By displaying output progressively, **streaming significantly   improves user experience**, particularly for longer responses  

As opposed to invoke(), which returns a single AIMessage after the model has finished generating its full response, stream() returns   multiple AIMessageChunk objects, each containing a portion of the output text. Importantly, each chunk in a stream is designed to be   gathered into a full message via summation  

```
from langchain.chat_models import init_chat_model

model = init_chat_model(model="llama3.1", model_provider="ollama", num_predict=100, temperature=0)

full = None

for chunk in model.stream("What is color of sun?"):
    full = chunk if full is None else full + chunk
    print(full.text)


print(full.content_blocks)


#output

The
The color
The color of
The color of the
The color of the Sun
The color of the Sun is
The color of the Sun is actually
The color of the Sun is actually white


```

###### Streaming events  
LangChain chat models can also stream semantic events using astream_events().  
Just like a frontend component has lifecycle events:  
  
onMount  
onUpdate  
onDestroy  

…an AI model in LangChain has semantic lifecycle events such as:  
on_start  
on_token    
on_thinking_start  
on_tool_start  
on_tool_end   
on_message  
on_end    
on_error  
  
These describe what stage of the process the model/agent is currently in.  

```
from langchain.chat_models import init_chat_model
import asyncio

model = init_chat_model(model="llama3.1", model_provider="ollama", num_predict=100, temperature=0)


async def myfunc():
    async for event in model.astream_events("Hello"):

        if event["event"] == "on_chat_model_start":
            print(f"Input {event['data']['input']}")
        elif event["event"] == "on_chat_model_stream":
            print(f"Token {event['data']['chunk'].text}")
        elif event["event"] == "on_chat_model_end":
            print(f"Full messages {event['data']['output'].text}")
        else:
            pass


if __name__ == "__main__":
    asyncio.run(myfunc())


Input Hello
Token Hello
Token !
Token  How
Token  are
Token  you
Token  today
Token ?
Token  Is
Token  there
Token  something
Token  I
Token  can
Token  help
Token  you
Token  with
Token  or
Token  would
Token  you
Token  like
Token  to
Token  chat
Token ?
Token 
Token 
Full messages Hello! How are you today? Is there something I can help you with or would you like to chat?

```

##### Batch
Batching a collection of independent requests to a model can significantly improve performance and reduce costs, as the processing can   be done in parallel:  

```
from langchain.chat_models import init_chat_model

model = init_chat_model(model="llama3.1", model_provider="ollama", num_predict=100, temperature=0)

responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])

for response in responses:
    print(response)
```

By default, batch() will only return the final output for the entire batch. If you want to receive the output for each individual  
input as it finishes generating, you can stream results with batch_as_completed():

```
from langchain.chat_models import init_chat_model

model = init_chat_model(model="llama3.1", model_provider="ollama", num_predict=100, temperature=0)

for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)

```
When processing a large number of inputs using batch() or batch_as_completed(), you may want to control the maximum number of parallel   calls. This can be done by setting the max_concurrency attribute in the RunnableConfig dictionary.  
```
model.batch(
    list_of_inputs,
    config={
        'max_concurrency': 5,  # Limit to 5 parallel calls
    }
)

```


```

A model with bind_tools() is tool-aware but not autonomous. The LLM sees the tools and can decide whether to call them, but YOU write the loop that decides what happens next. You have to manually check if a tool was called, execute it, pass the result back, and repeat until the model stops using tools.

An agent wraps this entire loop for you. You give it tools and a goal, and it automatically:
Side-by-Side Comparison
Task	            bind_tools()	       Agent
Model sees tools 	✅ Automatic	✅ Automatic
View tool calls  	✅ Automatic	✅ Automatic (internal)
Execute tools   	❌ Manual	    ✅ Automatic
Loop management 	❌ Manual	    ✅ Automatic
Error handling	  ❌ Manual	    ✅ Automatic (
State management	❌ Manual	    ✅ Automaticù


```


```

Decision Table
Need	Model + bind_tools()	Agent
Single tool call only	✅ Perfect	Overkill
Custom execution logic	✅ Full control	Limited
Multi-step reasoning	❌ Manual loop	✅ Automatic
Unknown tool count	❌ Error-prone	✅ Handles it
Middleware/hooks	❌ Not available	✅ Full support
Simplicity	❌ Code per tool call	✅ One call
Server-side tools	✅ Simple	✅ Works too


```














