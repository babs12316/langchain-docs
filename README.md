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












