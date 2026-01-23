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

Agent state is the data that flows through an agent's execution — it includes messages, custom fields, and any information the agent   needs to track during its lifecycle.  

Agent state is a DICTIONARY that contains:  
agent_state = {  
    "messages": [...],           # Conversation history (ALWAYS present)  
    "user_preferences": {...},   # Custom field (YOU define)  
    "conversation_count": 5,     # Custom field (YOU define)  
    "error_count": 0             # Custom field (YOU define)  
}  

Agent execution is the process of an agent running from start to finish — it receives input, makes decisions (using middleware, models, and tools), and produces output.


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

Differnce between result state and context

```
Use context for:
# Fixed user/system metadata passed at start
context=UserContext(
    user_role="admin",          # ← WHO is using it
    user_id="user_123",         # ← WHO identifier
    organization="TechCorp",    # ← Organization context
    security_level="high"       # ← Access level
)

# In @dynamic_prompt:
if context.user_role == "admin":
    # Admin-specific prompt
else:
    # User-specific prompt

Use state for:
# Conversation-specific data built during execution
{
    "messages": [...],              # Conversation history
    "request_type": "search",       # What type of request
    "session_id": "sess_123",       # Session tracking
    "error_count": 2,               # Error tracking
    "conversation_stage": 5         # Progress tracking
}

# In @dynamic_prompt:
request_type = state.get("request_type")
error_count = state.get("error_count")

if error_count > 3:
    # Error recovery prompt


Aspect	                                 Context          	      State
What it is	                              User metadata	           Conversation data
When passed	                              At invoke time (once)   	In invoke input
Can change	                               ❌ No (immutable)	      ✅ Yes (mutable)
Access in code	                          request.runtime.context	   request.state
Typical data	                            user_role, user_id, org    messages, request_type, error_count
Lifespan	                                 Entire conversation	       Single execution
Example usage	                             Permission checking	       Progress tracking



agent.invoke(                                              │
│      {                                                      │
│          "messages": [...],         ← STATE                │
│          "request_type": "search"   ← STATE                │
│      },                                                     │
│      context=UserContext(            ← CONTEXT             │
│          user_role="admin",                                │
│          user_name="Alice"                                 │
│      )                                                      │
│  )                           

```  


### Agent Invocation  
To invoke an agent update its state. All agents include a sequence of messages in their state; to invoke the agent, pass a new message:    
result = agent.invoke(  
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}  
)   

## Structured output  
In some situations, you may want the agent to return an output in a specific format.    

Structured output can be achived using ToolStrategy and ProviderStrategy  

ToolStrategy    
ToolStrategy uses artificial tool calling to generate structured output. This works with any model that supports tool calling:  

```
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')


```


ProviderStrategy  
ProviderStrategy uses the model provider’s native structured output generation. This is more reliable but only works with providers   that support native structured output (e.g., OpenAI):  

```
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```

## Memory
Agents maintain conversation history automatically through the message state.   
You can also configure the agent to use a custom state schema to remember additional information during the conversation.  

Information stored in the state can be thought of as the short-term memory of the agent:  
so short term memory = messages + custom_fields + any other field in state   

Custom state schemas must extend AgentState as a TypedDict.  

There are two ways to define custom state:  
Via middleware (preferred)  
Via state_schema on create_agent  

### Defining custom state  via middleware

```
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

# ========== 1. Custom State (extends AgentState) ==========
class CustomState(AgentState):
    user_preferences: dict  # ← Add your custom field

# ========== 2. Custom Middleware (extends AgentMiddleware) ==========
class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState    # ← Use custom state
    tools = [tool1, tool2]        # ← Define tools here

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # ← Custom logic before model call
        ...

# ========== 3. Create agent with middleware ==========
agent = create_agent(
    model,
    tools=tools,                    # Global tools
    middleware=[CustomMiddleware()] # Custom middleware
)

# ========== 4. Invoke with custom state ==========
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},  # ← Custom field
})
```

before_model and after_model are middleware methods that LangChain calls automatically at the right time — you don't call them   explicitly.  

### Defining custom state via state_schema
Use the state_schema parameter as a shortcut to define custom state that is only used in tools.


```
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})

```

so if custom state will be used in middelware then define custom state via middelware    
if custom state will be used in tool only then define custom state via state_schema    

### Streaming

We’ve seen how the agent can be called with invoke to get a final response. If the agent executes multiple steps, this may take a   while. To show intermediate progress, we can stream back messages as they occur. 

```
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values"):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

```

### Middelware  
Middleware provides powerful extensibility for customizing agent behavior at different stages of execution.  
You can use middleware to:  

Process state before the model is called (e.g., message trimming, context injection)    
Modify or validate the model’s response (e.g., guardrails, content filtering)  
Handle tool execution errors with custom logic
Implement dynamic model selection based on state or context    
Add custom logging, monitoring, or analytics   
```
# 1. Process State Before Model (Message Trimming)
from langchain.agents.middleware import before_model

@before_model
def trim_messages(state, runtime):
    """Keep only last 10 messages"""
    state["messages"] = state["messages"][-10:]
    return state

# 2. Modify/Validate Model Response (Guardrails)
@after_model
def validate_response(state, runtime):
    """Block unsafe content"""
    if "harmful" in state["messages"][-1].content:
        state["messages"][-1].content = "Sorry, I can't help with that."
    return state

# 3. Handle Tool Errors
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"Error: {e}", tool_call_id=request.tool_call["id"])

# 4. Dynamic Model Selection
@wrap_model_call
def select_model(request, handler):
    if len(request.messages) > 20:
        request.model = ChatOpenAI(model="gpt-4o")
    else:
        request.model = ChatOpenAI(model="gpt-4o-mini")
    return handler(request)

# 5. Custom Logging
@before_agent
def log_start(state, runtime):
    print(f"Agent started for user: {runtime.context.user_id}")
    return None

@after_agent
def log_end(state, runtime):
    print(f"Agent finished. Messages: {len(state['messages'])}")
    return None

# Usage
agent = create_agent(
    model=model,
    middleware=[
        trim_messages,
        validate_response,
        handle_errors,
        select_model,
        log_start,
        log_end
    ]
)
```
​
# Model  

LLMs are powerful AI tools that can interpret and generate text like humans. They’re versatile enough to write content, translate   languages, summarize, and answer questions without needing specialized training for each task.  

Model: Can see tools but needs manual execution.    
Agent: Can see + execute tools + manage state automatically.  

### Basic usage  
Models can be utilized in two ways:  
With agents - Models can be dynamically specified when creating an agent.  
Standalone - Models can be called directly (outside of the agent loop) for tasks like text generation, classification, or extraction   without the need for an agent framework.  


Initialize a model   
The easiest way to get started with a standalone model in LangChain is to use init_chat_model to initialize one from a chat model   provider of your choice   

```
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1")

response = model.invoke("Why do parrots talk?")
```

For init_chat_model you can pass following paramters  
model  string required  
The name or identifier of the specific model you want to use with a provider. You can also specify both the model and its provider in   a single argument using the ’:’ format, for example, ‘openai:o1’.  
​
api_key string  
The key required for authenticating with the model’s provider. This is usually issued when you sign up for access to the model. Often   accessed by setting an environment variable.  
​
temperature number  
Controls the randomness of the model’s output. A higher number makes responses more creative; lower ones make them more deterministic.  
​
max_tokens number  
Limits the total number of tokens in the response, effectively controlling how long the output can be.  
​
timeout number  
The maximum time (in seconds) to wait for a response from the model before canceling the request.  
​
max_retries number  
The maximum number of attempts the system will make to resend a request if it fails due to issues like network timeouts or rate limits.  

```
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)
```
**kwargs stands for keyword arguments.  
It lets a function accept any number of named arguments that weren’t explicitly listed in the function signature.  


Invocation  
A chat model must be invoked to generate an output. There are three primary invocation methods, each suited to different use cases.    

Invoke  
The most straightforward way to call a model is to use invoke() with a single message or a list of messages.  

```
response = model.invoke("Why do parrots have colorful feathers?")
print(response)

```


A list of messages can be provided to a chat model to represent conversation history. Each message has a role that models use to   indicate who sent the message in the conversation.  

```
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
from langchain.messages import HumanMessage, AIMessage, SystemMessage

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")

```

Stream  
Most models can stream their output content while it is being generated.  
By displaying output progressively, streaming significantly improves user experience, particularly for longer responses.  

Calling stream() returns an iterator that yields output chunks as they are produced.   
You can use a loop to process each chunk in real-time:  

```
for chunk in model.stream("Why do parrots have colorful feathers?"):
    print(chunk.text, end="|", flush=True)
```
As opposed to invoke(), which returns a single AIMessage after the model has finished generating its full response, stream() returns   multiple AIMessageChunk objects, each containing a portion of the output text. Importantly, each chunk in a stream is designed to be   gathered into a full message via summation:  


```
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

print(full.content_blocks)
# [{"type": "text", "text": "The sky is typically blue..."}]

```
Batch  
Batching a collection of independent requests to a model can significantly improve performance and reduce costs,  
as the processing can be done in parallel:  

```
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```
  
By default, batch() will only return the final output for the entire batch. If you want to receive the output for each individual   input as it finishes generating, you can stream results with batch_as_completed():  
Yield batch responses upon completion  

```
for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)

```

When processing a large number of inputs using batch() or batch_as_completed(), you may want to control   
the maximum number of parallel calls. This can be done by setting the max_concurrency attribute in the RunnableConfig dictionary.  

```

model.batch(
    list_of_inputs,
    config={
        'max_concurrency': 5,  # Limit to 5 parallel calls
    }
)
```

Tool calling   

Models can request to call tools that perform tasks such as fetching data from a database, searching the web, or running code.     
To make tools that you have defined available for use by a model, you must bind them using bind_tools.   
Some model providers offer built-in tools that can be enabled via model or invocation parameters (e.g. ChatOpenAI, ChatAnthropic).  

```
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model_with_tools = model.bind_tools([get_weather])  

response = model_with_tools.invoke("What's the weather like in Boston?")
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
```
Forcing tool calls  

By default, the model has the freedom to choose which bound tool to use based on the user’s input. However, you might want to force     choosing a tool, ensuring the model uses either a particular tool or any tool from a given list:    

```
model_with_tools = model.bind_tools([tool_1], tool_choice="any")

```

Force use of specific tool  
```
model_with_tools = model.bind_tools([tool_1], tool_choice="tool_1")
```


Parallel tool calls  

Many models support calling multiple tools in parallel when appropriate.    
This allows the model to gather information from different sources simultaneously.  
```
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke(
    "What's the weather in Boston and Tokyo?"
)


# The model may generate multiple tool calls
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]


# Execute all tools (can be done in parallel with async)
results = []
for tool_call in response.tool_calls:
    if tool_call['name'] == 'get_weather':
        result = get_weather.invoke(tool_call)
    ...
    results.append(result)

```

Structured output

Models can be requested to provide their response in a format matching a given schema. This is useful for ensuring the output can be easily parsed and used in subsequent   processing. LangChain supports multiple schema types and methods for enforcing structured output.  

 ```
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
```  

Message output alongside parsed structure  

It can be useful to return the raw AIMessage object alongside the parsed representation to access response metadata such as token counts.   
To do this, set include_raw=True when calling with_structured_output:  

```  
model_with_structure = model.with_structured_output(Movie, include_raw=True)

response
# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }

```    

Nested structures  

Schemas can be nested:    

```
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)
```  

# Prompt caching  
Many providers offer prompt caching features to reduce latency and cost on repeat processing of the same tokens. These features can be implicit or explicit:  
Implicit prompt caching: providers will automatically pass on cost savings if a request hits a cache. Examples: OpenAI and Gemini.  
Explicit caching: providers allow you to manually indicate cache points for greater control or to guarantee cost savings. Examples:  
ChatOpenAI (via prompt_cache_key)  
Anthropic’s AnthropicPromptCachingMiddleware  
Gemini.  
AWS Bedrock     

# Features to reduce latency and cost:

Feature       	                        
Parallel Tool Calls	Multiple tools run simultaneously	, 3 tools: 3s → 1s    
Streaming	Get tokens as they generate,	User sees response immediately  
Dynamic Model Selection	Use cheap/fast model when possible,	GPT-4o-mini for simple, GPT-4o for complex  
Caching	Reuse previous responses	Same query → instant result  
Prompt Compression	Shorter prompts ,faster	10k → 2k tokens  
Stateful Execution,	Reuse context	No repeating history   
Batch Processing	Process multiple at once	100 queries → 1 batch call  



Cost Reduction  
Feature	How It Helps	Savings  
Model Selection	Cheaper models for simple tasks	GPT-4o-mini ($0.15/M) vs GPT-4o ($2.50/M)  
Caching	Avoid repeat API calls	100% savings on cached  
Prompt Optimization	Fewer input tokens	50% input cost reduction  
Output Limits	max_tokens limits	Prevents long responses  
Batch API	Discounted pricing	25-50% batch discount  
Token Counting	Track usage	Avoid surprises    


Server side tool means using 3rd party tools such as web search
Server-Side Tools = Provider-Hosted  

```
# These are SERVER-SIDE tools:
# OpenAI hosts and runs them for you
{"type": "web_search"},      # OpenAI's web search
{"type": "code_interpreter"}, # OpenAI's Python interpreter
{"type": "file_search"}       # OpenAI's file search

# Provider: OpenAI, Anthropic, Google
# You: Just specify type string
# Provider: Executes everything

```  
Client-Side Tools = Your Tools

```
# These are CLIENT-SIDE tools:
@tool
def search_my_database(query: str) -> str:
    # YOU run this on your servers
    return db.query(query)

@tool
def check_inventory(item_id: str) -> str:
    # YOU run this
    return inventory.get(item_id)

```

Server-side tool use  
Some providers support server-side tool-calling loops: models can interact with web search, code interpreters, and other tools and analyze the results in a single conversational  turn.  

```
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4.1-mini")

tool = {"type": "web_search"}
model_with_tools = model.bind_tools([tool])

response = model_with_tools.invoke("What was a positive news story from today?")
response.content_blocks
```  


Rate limiting  
Many chat model providers impose a limit on the number of invocations that can be made in a given time period. If you hit a rate limit, you will typically receive a rate limit   error response from the provider, and will need to wait before making more requests.  
LangChain in comes with (an optional) built-in InMemoryRateLimiter.  
```
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 1 request every 10s
    check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
    max_bucket_size=10,  # Controls the maximum burst size. Burst size is the maximum number of requests allowed in a short time before rate limiting kicks in. max_bucket_size controls this.
)

model = init_chat_model(
    model="gpt-5",
    model_provider="openai",
    rate_limiter=rate_limiter  
)

```  


Log probabilities  
Log probabilities show how confident the model is about each token it generates — useful for debugging, confidence scoring, and analysis.  
Enabling this increases cost as additonal information is send
```
model = ChatOpenAI(
    model="gpt-4o",
    logprobs=True,      # Enable logprobs
)

response = model.invoke("What is 2+2?")

# Each token has log probabilities
for token, logprobs in response.logprobs:
    print(f"Token: '{token}'")
    print(f"Logprobs: {logprobs}")

Output

Token: 'The'
Logprobs: [{'token': 'The', 'logprob': -0.1}, ...]

Token: 'answer'
Logprobs: [{'token': 'answer', 'logprob': -0.05}, ...]

Token: 'is'
Logprobs: [{'token': 'is', 'logprob': -0.01}, ...]

Token: '4'
Logprobs: [{'token': '4', 'logprob': -0.001}, {'token': '5', 'logprob': -2.3}]


Your example:
Token: 'The'     logprob: -0.1   → 90.5% confidence (very confident)
Token: 'answer'  logprob: -0.05  → 95.1% confidence (extremely confident)
Token: 'is'      logprob: -0.01  → 99.0% confidence (almost certain)
Token: '4'       logprob: -0.001 → 99.9% confidence (almost certain)
Token: '5'       logprob: -2.3   → 10.0% confidence (very unlikely)

Logprob    Probability    Confidence    Meaning
  0.0           100%       Extremely high
 -0.1           90.5%        Very high
 -0.5           60.7%         Medium
 -1.0           36.8%          Low
 -2.0           13.5%       Very low
 -3.0            5.0%      Almost guess
 -5.0            0.7%      Pure guess

```


```
Confidence Scoring

def get_confidence(response):
    """Average logprob = model confidence"""
    avg_logprob = sum(logprob['logprob'] for token in response.logprobs 
                      for logprob in token.logprobs) / len(response.content)
    return avg_logprob

confidence = get_confidence(response)
if confidence < -0.5:
    print("Low confidence - ask for clarification")

```

```
Detect Hallucinations

def detect_uncertainty(response):
    """Find low-confidence tokens"""
    uncertain_tokens = []
    for token, logprobs in response.logprobs:
        top_logprob = logprobs[0]['logprob']
        if top_logprob < -1.0:  # Low confidence threshold
            uncertain_tokens.append(token)
    return uncertain_tokens

```


```
Filter Bad Responses


def filter_response(response):
    """Reject if too uncertain"""
    avg_confidence = sum(logprob['logprob'] for token in response.logprobs 
                         for logprob in token.logprobs) / len(response.content)
    
    if avg_confidence < -0.3:
        return None  # Retry
    return response

```

Invocation config  
When invoking a model, you can pass additional configuration through the config  
```
response = model.invoke(
    "Tell me a joke",
    config={
        "run_name": "joke_generation",      # Custom name for this run
        "tags": ["humor", "demo"],          # Tags for categorization
        "metadata": {"user_id": "123"},     # Custom metadata
        "callbacks": [my_callback_handler], # Callback handlers
    }
)
```  


Configurable models  
You can also create a runtime-configurable model by specifying configurable_fields. If you don’t specify a model value, then 'model' and 'model_provider' will be configurable by   default.  
```
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "gpt-5-nano"}},  # Run with GPT-5-Nano
)
configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},  # Run with Claude
)
```

# Messages

Messages are the fundamental unit of context for models in LangChain. They represent the input and output of models, carrying both the content and metadata needed to represent  
the state of a conversation when interacting with an LLM.  
Messages are objects that contain:  
 Role - Identifies the message type (e.g. system, user)  
 Content - Represents the actual content of the message (like text, images, audio, documents, etc.)  
 Metadata - Optional fields such as response information, message IDs, and token usage    

The simplest way to use messages is to create message objects and pass them to a model when invoking.    
```
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model("gpt-5-nano")

system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")

# Use with chat models
messages = [system_msg, human_msg]
response = model.invoke(messages)  # Returns AIMessage

```
 

Text prompts  
Text prompts are strings - ideal for straightforward generation tasks where you don’t need to retain conversation history.  
```
response = model.invoke("Write a haiku about spring")
```
Use text prompts when:    
You have a single, standalone request    
You don’t need conversation history    
You want minimal code complexity    

Message prompts  
Alternatively, you can pass in a list of messages to the model by providing a list of message objects.  
```
from langchain.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)

```

Use message prompts when:    
Managing multi-turn conversations    
Working with multimodal content (images, audio, files)     
Including system instructions     


Dictionary format  
You can also specify messages directly in OpenAI chat completions format.  
```
messages = [
    {"role": "system", "content": "You are a poetry expert"},
    {"role": "user", "content": "Write a haiku about spring"},
    {"role": "assistant", "content": "Cherry blossoms bloom..."}
]
response = model.invoke(messages)
```  

Message types  
 System message - Tells the model how to behave and provide context for interactions  
 Human message - Represents user input and interactions with the model  
 AI message - Responses generated by the model, including text content, tool calls, and metadata
 Tool message - Represents the outputs of tool calls   


 System message  
A SystemMessage represent an initial set of instructions that primes the model’s behavior. You can use a system message to set the tone, define the model’s role, and establish     guidelines for responses.    
```
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)

```
Detailed Message 
​
```
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)

```

Human message  
A HumanMessage represents user input and interactions. They can contain text, images, audio, files, and any other amount of multimodal content.  
```
# Using a string is a shortcut for a single HumanMessage
response = model.invoke("What is machine learning?")
```

Message Metadata  
```
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)

```

AI message  
An AIMessage represents the output of a model invocation. They can include multimodal data, tool calls, and provider-specific metadata that you can later access.  
```
response = model.invoke("Explain AI")
print(type(response))  # <class 'langchain.messages.AIMessage'>
```

Providers weigh/contextualize types of messages differently, which means it is sometimes helpful to manually create a new AIMessage object and insert it into the message history   as if it came from the model.  
```
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# Create an AI message manually (e.g., for conversation history)
ai_msg = AIMessage("I'd be happy to help you with that question!")

# Add to conversation history
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # Insert as if it came from the model
    HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)

```



Tool calls  
When models make tool calls, they’re included in the AIMessage:  

```
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

def get_weather(location: str) -> str:
    """Get the weather at a location."""
    ...

model_with_tools = model.bind_tools([get_weather])
response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"ID: {tool_call['id']}")

```

Token usage  
An AIMessage can hold token counts and other usage metadata in its usage_metadata field:

```
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")

response = model.invoke("Hello!")
response.usage_metadata


{'input_tokens': 8,
 'output_tokens': 304,
 'total_tokens': 312,
 'input_token_details': {'audio': 0, 'cache_read': 0},
 'output_token_details': {'audio': 0, 'reasoning': 256}}


```


Streaming and chunks    
LangChain's model.stream() yields AIMessageChunk objects incrementally, which you accumulate using + operator to reconstruct the full   AIMessage.  

Unlike model.invoke() that waits for complete response and returns one AIMessage, streaming gives you small AIMessageChunk pieces as   the model generates tokens. Each chunk contains partial text (or content_blocks), and you merge them progressively to build the final   message.  

The code collects chunks in a list while maintaining a running full_message by using the + operator (which calls __add__ on   AIMessageChunk to merge content, tool_calls, etc.):  

```
from langchain_core.messages import AIMessageChunk

chunks = []          # Optional: store all chunks if needed later
full_message = None  # Accumulator for merged response

for chunk in model.stream("What color is the sky?"):
    chunks.append(chunk)
    print(chunk.text)  # Print incrementally: "The", "The sky", etc.
    
    # Merge: first chunk becomes base, subsequent chunks add to it
    full_message = chunk if full_message is None else full_message + chunk

# After loop: full_message is complete AIMessage-equivalent
print(full_message.text)
# "The sky is typically blue on a clear day."
```

ToolMessage
ToolMessage is the "answer back" you send to the AI after it asks you to run a tool.  

Here's the simple flow:  

AI says: "Hey, run the search tool with 'iPhone'"  
This creates an AIMessage with tool_calls=[{name: "search", id: "abc123"}]  
You run the tool: search("iPhone") → ["iPhone 15", "iPhone 14"]
You reply with ToolMessage:   
```
ToolMessage(
    content="['iPhone 15', 'iPhone 14']",  # Tool result
    tool_call_id="abc123"                  # Matches AI's request exactly!
)
```

Message content  
You can think of a message’s content as the payload of data that gets sent to the model. Messages have a content attribute that is   loosely-typed, supporting strings and lists of untyped objects (e.g., dictionaries). This allows support for provider-native   structures directly in LangChain chat models, such as multimodal content and other data.  

Standard content blocks  
LangChain provides a standard representation for message content that works across providers.  
Message objects implement a content_blocks property that will lazily parse the content attribute into a standard, type-safe   representation.     

Content blocks were introduced as a new property on messages in LangChain v1 to standardize content formats across providers while   maintaining backward compatibility with existing code.  
Content blocks are not a replacement for the content property, but rather a new property that can be used to access the content of a    message in a standardized format.    

# Tools

Tools extend what agents can do—letting them fetch real-time data, execute code, query external databases, and take actions in the   world.  
Under the hood, tools are callable functions with well-defined inputs and outputs that get passed to a chat model. The model decides   when to invoke a tool based on the conversation context, and what input arguments to provide.    

Create tools  
​
Basic tool definition  
The simplest way to create a tool is with the @tool decorator. By default, the function’s docstring becomes the tool’s description   that helps the model understand when to use it:  

```
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

Type hints are required as they define the tool’s input schema. The docstring should be informative and concise to help the model   understand the tool’s purpose.  


Customize tool properties  
​
Custom tool name  
By default, the tool name comes from the function name. Override it when you need something more descriptive:  

```
@tool("web_search")  # Custom name
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.name)  # web_search

```

Custom tool description   
Override the auto-generated tool description for clearer model guidance:  
```
@tool("calculator", description="Performs arithmetic calculations. Use this for any math problems.")
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))
```


ToolRuntime  
ToolRuntime gives tools access to "extra info" like user data, memory, and conversation state without the LLM seeing it.   

ToolRuntime: A unified parameter that provides tools access to state, context, store, streaming, config, and tool call ID.  

Accessing state:  
Tools can access the current graph state using ToolRuntime:  
```
from langchain.tools import tool, ToolRuntime

# Access the current conversation state
@tool
def summarize_conversation(
    runtime: ToolRuntime
) -> str:
    """Summarize the conversation so far."""
    messages = runtime.state["messages"]

    human_msgs = sum(1 for m in messages if m.__class__.__name__ == "HumanMessage")
    ai_msgs = sum(1 for m in messages if m.__class__.__name__ == "AIMessage")
    tool_msgs = sum(1 for m in messages if m.__class__.__name__ == "ToolMessage")

    return f"Conversation has {human_msgs} user messages, {ai_msgs} AI responses, and {tool_msgs} tool results"

# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")

```

Updating state:  
Use Command to update the agent’s state or control the graph’s execution flow:  
```
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

# Update the conversation history by removing all messages
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})

```


Context  
Access immutable configuration and contextual data like user IDs, session details, or application-specific configuration through   runtime.context.  
Tools can access runtime context through ToolRuntime:   

```
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)

```


Memory (Store)  
Access persistent data across conversations using the store. The store is accessed via runtime.store and allows you to save and   retrieve user-specific or application-specific data.  
Tools can access and update the store through ToolRuntime:  

```
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev


```


Stream writer  
Stream custom updates from tools as they execute using runtime.stream_writer. This is useful for providing real-time feedback to  
users about what a tool is doing.  

```

from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
```

# Context window  
Context window is the max amount of text (measured in tokens) an AI model can "see" and process at once.  

Short term memory
Chat models accept context using messages, which include instructions (a system message) and inputs (human messages). In chat   applications, messages alternate between human inputs and model responses, resulting in a list of messages that grows longer over   time. Because context windows are limited, many applications can benefit from using techniques to remove or “forget” stale information.  


To add short-term memory (thread-level persistence) to an agent, you need to specify a checkpointer when creating an agent.      

A thread organizes multiple interactions in a session, similar to the way email groups messages in a single conversation.  
```
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")
agent = create_agent(
    model,
    tools=[],  # Add your tools here
    checkpointer=InMemorySaver(),  # ← Turn on memory!
)

# Conversation 1
result1 = agent.invoke(
    {"messages": [HumanMessage(content="Hi! My name is Bob")]},
    {"configurable": {"thread_id": "1"}}  # ← Conversation ID 1
)
print("Msg 1:", result1["messages"][-1].content)

# Conversation 2 (SAME thread_id → remembers!)
result2 = agent.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    {"configurable": {"thread_id": "1"}}  # ← Same ID = remembers Bob!
)
print("Msg 2:", result2["messages"][-1].content)
```


```
Short-term memory location:
InMemorySaver() → Python RAM (temp)  
SqliteSaver() → local file (persistent)  
PostgresSaver() → database (production)

thread_id → Memory slot name (like a folder)

```

Customizing agent memory  
By default, agents use AgentState to manage short term memory, specifically the conversation history via a messages key.  
You can extend AgentState to add additional fields. Custom state schemas are passed to create_agent using the state_schema parameter.  

# context , agent state and store are used to provide infomration to tool   
LLM calls tool → LangChain injects ToolRuntime → Tool gets context/state/store → Tool uses data
Inside Every Tool  

```
@tool
def my_smart_tool(arg1: str, runtime: ToolRuntime):  # runtime = MAGIC BOX
    user_id = runtime.context.user_id           # Fixed config
    messages = runtime.state["messages"]        # Chat history  
    prefs = runtime.store.get(("users", user_id)) # Permanent memory
    # Now tool is SUPER SMART! 🧠

```


```
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,  
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",  
        "preferences": {"theme": "dark"}  
    },
    {"configurable": {"thread_id": "1"}})

```
```
Where	                           Lifetime                      	Mutability	                          Use When                   	
Context (runtime.context)     	Single run	                     Immutable (read-only)	           Fixed config at start of chat	user_id, api_key, model_name
Agent State (runtime.state)	    Single thread (conversation)	    Mutable (changes during chat)   	Short-term memory during one chat	messages, conversation_history, temp_results
Store (runtime.store)	           Forever (cross conversations)	   Mutable (update anytime)	      Long-term memory across ALL chats	user_preferences, knowledge_base, research_notes


Fixed at chat start?    → Context
Changes during 1 chat?  → Agent State  
Lasts forever?          → Store

👤 user_id              → Context
💬 chat messages        → Agent State  
🧠 user preferences     → Store

```

Customizing agent memory  
By default, agents use AgentState to manage short term memory, specifically the conversation history via a messages key.  
You can extend AgentState to add additional fields. Custom state schemas are passed to create_agent using the state_schema parameter.  

```
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver


class CustomAgentState(AgentState):  
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,  
    checkpointer=InMemorySaver(),
)

# Custom state can be passed in invoke
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",  
        "preferences": {"theme": "dark"}  
    },
    {"configurable": {"thread_id": "1"}})

```

With short-term memory enabled, long conversations can exceed the LLM’s context window. Common solutions are:    
Most LLMs have a maximum supported context window (denominated in tokens).  
One way to decide when to truncate messages is to count the tokens in the message history and truncate whenever it approaches that   limit. If you’re using LangChain, you can use the trim messages utility and specify the number of tokens to keep from the list, as   well as the strategy (e.g., keep the last max_tokens) to use for handling the boundary.  
To trim message history in an agent, use the @before_model middleware decorator:  

```
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    your_model_here,
    tools=your_tools_here,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""

```


Delete messages  
You can delete messages from the graph state to manage the message history.  
This is useful when you want to remove specific messages or clear the entire message history.  
To delete messages from the graph state, you can use the RemoveMessage.  

To remove specific messages:

```
from langchain.messages import RemoveMessage  

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

```

To remove all messages:  

```
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```


Summarize messages  
The problem with trimming or removing messages, as shown above, is that you may lose information from culling of the message queue.   Because of this, some applications benefit from a more sophisticated approach of summarizing the message history using a chat model.  

```
# ========================================
# SMART CHAT AGENT WITH AUTO-SUMMARIZATION
# ========================================
from langchain.agents import create_agent                    # Main agent builder
from langchain.agents.middleware import SummarizationMiddleware  # Auto-compress long chats
from langgraph.checkpoint.memory import InMemorySaver       # RAM-based memory (lost on restart)
from langchain_core.runnables import RunnableConfig         # Config type for thread memory


# 1. Create memory storage (short-term chat history)
checkpointer = InMemorySaver()                               # Stores conversation state in RAM


# 2. Build agent with summarization + memory
agent = create_agent(
    model="gpt-4o",                                          # Main AI model (brain)
    tools=[],                                                # No tools needed for this demo
    middleware=[                                             # Special behaviors!
        SummarizationMiddleware(                             # MAGIC: Auto-summarize long chats
            model="gpt-4o-mini",                             # Cheap/fast summarizer model
            trigger=("tokens", 4000),                        # Summarize when chat > 4k tokens
            keep=("messages", 20)                            # Keep last 20 messages FULL (recent context)
        )
    ],
    checkpointer=checkpointer,                               # Enable memory (REQUIRED!)
)

# 3. Config for SINGLE conversation (thread_id = memory key)
config: RunnableConfig = {"configurable": {"thread_id": "1"}}  # All messages use SAME memory slot!

# 4. Message 1: Introduce self (SAVED to thread "1")
agent.invoke({"messages": "hi, my name is bob"}, config)     # Memory: ["Hi Bob"]

# 5. Message 2: Cat poem (LOADS memory → knows you're Bob)
agent.invoke({"messages": "write a short poem about cats"}, config)  
# Memory: ["Hi Bob", "Cat poem"]

# 6. Message 3: Dog poem (LOADS full history)
agent.invoke({"messages": "now do the same but for dogs"}, config)
# Memory: ["Hi Bob", "Cat poem", "Dog poem"]
# Tokens < 4000 → No summarization yet

# 7. Message 4: Test memory! (LOADS entire history → finds name)
final_response = agent.invoke({"messages": "what's my name?"}, config)
# AI sees: ["Hi Bob", "Cat poem", "Dog poem"] → "Your name is Bob!"

# 8. Print final answer (AI REMEMBERED after 3 messages!)
final_response["messages"][-1].pretty_print()                # Shows: "Your name is Bob!"
```
# Access memory  
You can access and modify the short-term memory (state) of an agent in several ways:  
Read short-term memory in a tool  

```
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)
# > User is John Smith.

```

Write short-term memory from tools  
To modify the agent’s short-term memory (state) during execution, you can return state updates directly from the tools.  
This is useful for persisting intermediate results or making information accessible to subsequent tools or prompts.  

```
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):  
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={  
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState, 
    context_schema=CustomContext,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
```



Prompt (Dynamic)  
Access short term memory (state) in middleware to create dynamic prompts based on conversation history or custom state fields.  

```
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class CustomContext(TypedDict):
    user_name: str


def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()

```


Before model  
Access short term memory (state) in @before_model middleware to process messages before model calls.   
```
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


agent = create_agent(
    "gpt-5-nano",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""

```

Differncen betwwen before_model and middleware  

Middelware called Once per invoke() and before_model call before every LLM call (agent loop repeats)

```
class DebugMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):      # ONCE per invoke()
        print("🚀 AGENT STARTING (1x)")
        return state
    
    @before_model                               # MANY times!
    def before_model_hook(self, state, runtime):
        print(f"🤖 AI THINKING #{len(state['messages'])}")
        return state

agent = create_agent(model, tools, middleware=[DebugMiddleware()])

agent.invoke("Complex task with tools")  # Triggers 3 AI loops

```

After model  
Access short term memory (state) in @after_model middleware to process messages after model calls.  
@after_model runs after EVERY AI call (same frequency as @before_model!)
```
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)

```

# Streaming  
Stream real-time updates from agent runs  

LangChain implements a streaming system to surface real-time updates.  
Streaming is crucial for enhancing the responsiveness of applications built on LLMs. By displaying output progressively, even before a     complete response is ready, streaming significantly improves user experience (UX), particularly when dealing with the latency of   LLMs.


Supported stream modes   
Pass one or more of the following stream modes as a list to the stream or astream methods:  

updates: 	Streams state updates after each agent step. If multiple updates are made in the same step (e.g., multiple nodes are run),   those updates are streamed separately.  
messages:	stream tokens as they are produced by the LLM  
custom:	Streams custom data from inside your graph nodes using the stream writer. 


Agent Progress  
To stream agent progress, use the stream or astream methods with stream_mode="updates". This emits an event after every agent step.
```
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)
for chunk in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")


step: model
content: [{'type': 'tool_call', 'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_OW2NYNsNSKhRZpjW0wm2Aszd'}]

step: tools
content: [{'type': 'text', 'text': "It's always sunny in San Francisco!"}]

step: model
content: [{'type': 'text', 'text': 'It's always sunny in San Francisco!'}]
```
​
LLM tokens  
To stream tokens as they are produced by the LLM, use stream_mode="messages".  
```
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)
for token, metadata in agent.stream(  
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")

```

Custom updates  
To stream updates from tools as they are executed, you can use get_stream_writer.  
```
from langchain.agents import create_agent
from langgraph.config import get_stream_writer  


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()  
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"
):
    print(chunk)


Looking up data for city: San Francisco
Acquired data for city: San Francisco

```

Streaming from sub-agents  
When there are multiple LLMs at any point in an agent, it’s often necessary to disambiguate the source of messages as they are   generated.  
To do this, pass a name to each agent when creating it. This name is then available in metadata via the lc_agent_name key when   streaming in "messages" mode.  

```
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


weather_model = init_chat_model("openai:gpt-5.2")
weather_agent = create_agent(
    model=weather_model,
    tools=[get_weather],
    name="weather_agent",  
)


def call_weather_agent(query: str) -> str:
    """Query the weather agent."""
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].text


supervisor_model = init_chat_model("openai:gpt-5.2")
agent = create_agent(
    model=supervisor_model,
    tools=[call_weather_agent],
    name="supervisor",  
)

```


Disable streaming  
In some applications you might need to disable streaming of individual tokens for a given model. This is useful when:  
Working with multi-agent systems to control which agents stream their output  
Mixing models that support streaming with those that do not  
Deploying to LangSmith and wanting to prevent certain model outputs from being streamed to the client  
Set streaming=False when initializing the model.  

```
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    streaming=False
)


```

# Structured output

Structured output allows agents to return data in a specific, predictable format. Instead of parsing natural language responses, you get structured data in the form of JSON   objects, Pydantic models, or dataclasses that your application can use directly.     
 
Response format  
Use response_format to control how the agent returns structured data:  
ToolStrategy[StructuredResponseT]: Uses tool calling for structured output.  Its Universal: Works with any tool-calling model
ProviderStrategy[StructuredResponseT]: Uses provider-native structured output. Its Faster: Single LLM call, no extra tool execution  but Provider-specific (only OpenAI, Anthropic, Gemini, Grok, etc.)  
type[StructuredResponseT]: Schema type - automatically selects best strategy based on model capabilities  
None: Structured output not explicitly requested   

Provider Stratergy

```
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents.structured_output import ProviderStrategy

class ContactInfo(BaseModel):
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

model = ChatOpenAI(model="gpt-4o")  # Native support
agent = create_agent(
    model=model,
    tools=[],  # No real tools needed
    response_format=ProviderStrategy(ContactInfo),  # Native API
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})
print(result.structured_response)
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')

```  


ToolStrategy (Tool Calling)  

```
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),  # Tool calling only
    tools=[],
    response_format=ToolStrategy(
        ContactInfo,
        tool_message_content="Contact info extracted!",  # Custom chat message
        handle_error=True  # Auto-retry validation errors
    ),
)

```

# Middelware

Middleware provides a way to more tightly control what happens inside the agent. Middleware is useful for the following:  
Tracking agent behavior with logging, analytics, and debugging.  
Transforming prompts, tool selection, and output formatting.  
Adding retries, fallbacks, and early termination logic.  
Applying rate limits, guardrails, and PII detection.  


wrap_model_call   
Wraps every LLM call in the agent loop.  
• Dynamic model swapping (gpt-4o → o1 based on complexity)  
• Request modification (add context/tools)  
• Retry logic (rate limits/timeouts)  
• Caching (same prompt → cached response)  
• Observability (log inputs/outputs)  


wrap_tool_call
Wraps every individual tool execution   
• Custom error handling (return safe message vs crash)   
• Tool permissions (block sensitive tools)  
• Mocking (emulate tools for testing)  
• Logging/caching tool calls  
• Rate limiting per-tool     


# Runtime information

context - contain read only info such as user id and is used for personalization  

Store: a BaseStore instance used for long-term memory  

Stream writer: an object used for streaming information via the "custom" stream mode  


You can access the runtime information within tools and middleware.  

Inside middleware  
You can access runtime information in middleware to create dynamic prompts, modify messages, or control agent behavior based on user context.   
Use request.runtime to access the Runtime object inside middleware decorators. The runtime object is available in the ModelRequest parameter passed to middleware functions.    


# Why do agents fail?  
When agents fail, it’s usually because the LLM call inside the agent took the wrong action / didn’t do what we expected. LLMs fail for one of two reasons:  
The underlying LLM is not capable enough  
The “right” context was not passed to the LLM    

Context engineering is providing the right information and tools in the right format so the LLM can accomplish a task.  

The agent loop  
A typical agent loop consists of two main steps:  
Model call - calls the LLM with a prompt and available tools, returns either a response or a request to execute tools  
Tool execution - executes the tools that the LLM requested, returns tool results    

This loop continues until the LLM decides to finish.  


To build reliable agents, you need to control what happens at each step of the agent loop, as well as what happens between steps.   

Model Context  
What goes into model calls (instructions, message history, tools, response format)  
Transient - Scope/saved for : Single invoke(), saved in state, can access using request.state  

Tool Context  
What tools can access and produce (reads/writes to state, store, runtime context)  
Persistent- scope/saved for : Multiple invoke()s, saved in store/checkpoints, accessed using runtime.store/ checkpoints  

Life-cycle Context  
What happens between model and tool calls (summarization, guardrails, logging, etc.)
Persistent- scope/saved for : Multiple invoke()s, saved in store/checkpoints, accessed using runtime.store/ checkpoints    

Tools let the model interact with databases, APIs, and external systems. 


​
