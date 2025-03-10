from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# 从本地服务创建模型
model = ChatOpenAI(
    #model_name="deepseek-r1:32b",
    model_name="qwen2",
    openai_api_base="http://127.0.0.1:11434/v1",
    openai_api_key="EMPTY",
    streaming=False
)
"""
得到的模型model内容如下：
ChatOpenAI(client=<openai.resources.chat.completions.completions.Completions object at 0x77d780837c70>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x77d780869d80>, root_client=<openai.OpenAI object at 0x77d780da9e10>, root_async_client=<openai.AsyncOpenAI object at 0x77d780837cd0>, model_name='qwen2', model_kwargs={}, openai_api_key=SecretStr('**********'), openai_api_base='http://127.0.0.1:11434/v1')
"""


# 手动构建消息列表
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]
"""
得到的消息列表message内容如下：
[SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]
"""

# model.invoke最终会调用BaseChatModel的generate函数生成response
response = model.invoke(messages)
#response = model.invoke("Hello")
#response = model.invoke([{"role": "user", "content": "Hello"}])
#response = model.invoke([HumanMessage("Hello")])
print(response.content)
"""
得到的response内容如下：
AIMessage(content='Ciao!', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 4, 'prompt_tokens': 22, 'total_tokens': 26, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'qwen2', 'system_fingerprint': 'fp_ollama', 'finish_reason': 'stop', 'logprobs': None}, id='run-607f6342-a391-4b9f-94a3-fda689d5a10d-0', usage_metadata={'input_tokens': 22, 'output_tokens': 4, 'total_tokens': 26, 'input_token_details': {}, 'output_token_details': {}})
"""


# model.stream最终会调用BaseChatModel的_stream函数一次生成一个token。
for token in model.stream(messages):
    print(token.content, end="|")
print("\n")


# 通过prompt template创建消息列表
system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
prompt.to_messages()

response = model.invoke(prompt)
print(response.content)