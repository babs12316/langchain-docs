# langchain-docs

You can declare model name in create_agent using model identifier string. Its a string that follows the format `provider:model` (e.g. openai:gpt-5)

from langchain.agent import create_agent
agent= create_agent(model="ollama:llm2", tools= tools) 
