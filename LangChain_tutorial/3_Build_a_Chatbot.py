from langchain_openai.chat_models import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
import asyncio

# 基于本地服务创建模型model
model = ChatOpenAI(
    #model_name="deepseek-r1:32b",
    model_name="qwen2",
    openai_api_base="http://127.0.0.1:11434/v1",
    openai_api_key="EMPTY",
    streaming=False
)

# 定义StateGraph对象
workflow = StateGraph(state_schema=MessagesState)

# 后面我们定义了一个node "model", 该node对应执行函数call_model定义如下：
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# 定义一个node "model"，以及一条边
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()  # output包含state中的所有消息，这里我们只打印最后一条消息。

query = "What's my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()


# 切换到thread "abc234"，历史消息将重新开始记录。
config = {"configurable": {"thread_id": "abc234"}}
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()


# 切换回thread "abc123"，原来的历史消息仍然存在。
config = {"configurable": {"thread_id": "abc123"}}
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

print("*******************************************************")

#对于异步支持，更新 call_model 节点使其成为一个异步函数（使用async关键字），并在调用应用程序时使用.ainvoke
#这样可以确保在需要异步处理的场景中，比如与外部服务通信或执行长时间运行的任务时，能够更高效地进行操作。通过这种方式，您可以利用异步编程的优势来提高应用程序的响应速度和性能。

# Async function for node:
async def call_model(state: MessagesState):
    response = await model.ainvoke(state["messages"])
    return {"messages": response}

# Define graph as before:
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())

async def test():
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

asyncio.run(test())

print("*******************************************************")

"""
到目前为止，我们所做的只是在模型周围添加了一个简单的持久化层。我们可以通过引入提示模板使聊天机器人变得更加复杂和个性化。这样做可以让我们根据用户的具体需求定制对话体验，比如通过动态调整提示内容来引导对话方向或提供更相关的回复。

提示模板用于将原始用户输入转换成LLM可以处理的格式。在这个例子中，原始用户输入只是一个消息，我们将其传递给LLM。  
现在让我们使这个过程变得更复杂一些。首先，添加一个带有自定义指令的系统消息（但仍以消息作为输入）。接下来，除了消息之外，我们还会增加更多的输入。

为了添加系统消息，我们将创建一个ChatPromptTemplate。我们将使用MessagesPlaceholder来传递所有的消息。具体来说：

我们会创建一个包含系统消息的ChatPromptTemplate，该系统消息含有定制的指令。
利用MessagesPlaceholder，我们可以将所有对话消息传递进去，以便于在提示中使用。

这样做的目的是为了让聊天机器人不仅能够处理简单的消息输入，还能够基于更复杂的上下文和指令进行回应，从而提供更加个性化和精确的回复。
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

"""
我们现在可以更新我们的应用程序以包含这个模板：

通过将上述提示模板集成到应用程序中，我们可以使聊天机器人的对话能力更加复杂和个性化。这意味着，当用户与聊天机器人交互时，不仅可以处理简单的消息输入，还能基于预先设定的系统消息和动态输入内容生成更丰富、更相关的回复。这样做能够显著提升用户体验，使得对话更加自然流畅。具体实现时，需要确保将用户输入以及任何必要的额外信息正确地传递给提示模板，并通过该模板格式化后提供给语言模型。
"""

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

"""我们以相同的方式调用应用程序："""

config = {"configurable": {"thread_id": "abc345"}}
query = "Hi! I'm Jim."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

"""太棒了！现在让我们使提示变得更加复杂一些。"""
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

"""注意，我们已经在提示中添加了一个新的language输入。我们的应用程序现在有两个参数——messages和language。我们应该更新我们应用程序的状态以反映这一点："""
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc456"}}
query = "Hi! I'm Bob."
language = "Spanish"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

"""注意，整个状态会被持久化，因此如果不需要更改，我们可以省略像language这样的参数："""
query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages},
    config,
)
output["messages"][-1].pretty_print()

"""
管理对话历史  
在构建聊天机器人时，一个重要的概念是如何管理对话历史。如果不对对话历史进行管理，消息列表将无限制地增长，可能会超出LLM的上下文窗口。因此，添加一个步骤来限制传递的消息大小是很重要的。
"""

"""
重要的是，你需要在应用提示模板之前，并且在从消息历史中加载先前消息之后进行这一步操作。
我们可以通过在提示前添加一个简单的步骤来适当地修改messages key，然后将这个new chain包裹在消息历史类中来实现这一点。
LangChain提供了一些内置的帮助函数来管理消息列表。在这个例子中，我们将使用trim_messages帮助函数来减少发送给模型的消息数量。这个trimmer允许我们指定想要保留多少tokens，以及其他参数，如是否总是保留系统消息以及是否允许部分消息：
"""

"""
token_counter设置为model，会出现get_num_tokens_from_messages() is not presently implemented for model cl100k_base错误。 
这是因为我们使用的模型qwen2不支持token个数计算。
一个解决方法是自定义token计数器（即下面的custom_token_counter函数），使用OpenAI的tiktoken库来计算token数量。
"""

from langchain_core.messages import SystemMessage, trim_messages
import tiktoken

def custom_token_counter(messages):
    encoding = tiktoken.get_encoding("cl100k_base")  # 使用 cl100k_base 编码
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message.content))
    return total_tokens

trimmer = trim_messages(
    max_tokens=35,
    strategy="last",
    token_counter=custom_token_counter,  # 使用自定义的 token 计数器,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmer.invoke(messages)

"""为了在我们的chain中使用它，我们只需要在将message传递给prompt之前运行trimmer。"""
workflow = StateGraph(state_schema=State)

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

"""现在，如果我们尝试问模型我们的名字，它将不知道，因为我们修剪了聊天历史的那部分"""
config = {"configurable": {"thread_id": "abc567"}}
query = "What is my name?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

"""但如果我们就最近几条消息中的信息进行提问，它会记得"""
config = {"configurable": {"thread_id": "abc678"}}
query = "What math problem did I ask?"
language = "English"

input_messages = messages + [HumanMessage(query)]
output = app.invoke(
    {"messages": input_messages, "language": language},
    config,
)
output["messages"][-1].pretty_print()

"""
流式传输  
现在我们已经拥有一个可以正常工作的聊天机器人。然而，对于聊天机器人应用程序来说，一个非常重要的用户体验考虑是流式传输。由于LLM有时需要一些时间来生成回复，大多数应用程序为了提升用户体验，会将每个生成的token即时流式传输回用户端。这使得用户能够看到进度。
"""

"""
默认情况下，LangGraph应用程序中的.stream方法会进行流式传输。设置stream_mode="messages"可以让我们改为流式传输输出的tokens：
"""

config = {"configurable": {"thread_id": "abc789"}}
query = "Hi I'm Todd, please tell me a joke."
language = "English"

input_messages = [HumanMessage(query)]
for chunk, metadata in app.stream(
    {"messages": input_messages, "language": language},
    config,
    stream_mode="messages",
):
    if isinstance(chunk, AIMessage):  # Filter to just model responses
        print(chunk.content, end="|")

print("\n*********************************************")