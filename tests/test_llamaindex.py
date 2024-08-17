from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage 
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core import Settings
from typing import List

MAX_CONTEXT = 32000

llm = OpenAILike(
        model="zskalo/gemma-1.1-2b-it-rag-sft",
        api_key="my-token",
        api_base="http://10.42.47.11:8000/v1",
        max_tokens=0.2*MAX_CONTEXT
    )

Settings.llm = llm
token_limit=int(MAX_CONTEXT*0.8) # 80% of maximum tokens the model can handle
chat_engine = SimpleChatEngine.from_defaults(
    #system_prompt="You are a Q&A machine!",
    memory=ChatMemoryBuffer.from_defaults(token_limit=token_limit),
)

init_msg = ChatMessage(role= "user", content="Tell me about your abilities!")
print(f"{init_msg.role.value}: {init_msg.content}")

response = chat_engine.chat(init_msg.content)
print(f"assistant: {response}")

while True:
    user_input = input("user: ")
    
    response = chat_engine.stream_chat(user_input)
    print("assistant: ", end="" )
    for token in response.response_gen:
        print(token, end="")
    print("")
