
# Langgraph AI Agents

This projects inludes custom AI agents built using Python's Langgraph Framework.




## Usage/Examples

To use the the custom agents, plug in your own local chat model in the function call

```python
ChatOllama(model='your_model_name',temperature=0).bind_tools([searxng_search])
```
, also modify the provider according to your needs. 

By default the agents are stateless, once exited from the program run time, the conversation hitory is lost. 

To start an agent, run:
```
Python3 file_name.py
```


## Installation

To run the agents, you need `langchain`, `langchain_core`, `langchain_ollama`, `requests`
modules installed 
```bash
  pip install package-name
```
    