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














