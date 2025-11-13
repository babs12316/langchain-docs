# langchain-docs

## Agents      
Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use, and iteratively work towards solutions. 
  
Agents takes model, tools and system_prompt  

Lets check these component one by one.  

### Model  
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





