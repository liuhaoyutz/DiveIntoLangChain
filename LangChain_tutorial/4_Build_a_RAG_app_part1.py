from langchain_openai.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# 基于本地服务创建llm
llm = ChatOpenAI(
    #model_name="deepseek-r1:32b",
    model_name="qwen2",
    openai_api_base="http://127.0.0.1:11434/v1",
    openai_api_key="EMPTY",
    streaming=False
)

# 初始化Hugging Face嵌入模型
embeddings = HuggingFaceEmbeddings()

"""
可选的vector store包括：  
In-memory, AstraDB, Chroma, FAISS, Milvus, MongoDB, PGVector, Pinecone, Qdrant  

使用不同的vector store需要安装不同的库，并进行相应配置。  
下面我们使用的是最简单的In-memory方式。  
"""
# 创建向量数据库，注意参数指定了关联的embedding模型。
vector_store = InMemoryVectorStore(embeddings)

# 创建web数据加载器并加载指定web页面的内容到docs。
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 创建文本分割器将docs内容分割成大小为1000，有200 overlap的块，结果保存在all_splits中。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# 将文本块使用embedding模型向量化，然后保存到向量数据库vector_store中。
_ = vector_store.add_documents(documents=all_splits)

# 从LangChain hub获取一个RAG相关的prompt template。
prompt = hub.pull("rlm/rag-prompt")

"""
状态  
应用程序的状态控制哪些数据被输入到应用程序中，哪些数据在步骤之间传递，以及哪些数据作为输出。它通常是一个TypedDict，也可以是一个Pydantic的BaseModel。  
对于一个简单的RAG应用程序，我们可以只跟踪输入的问题、检索到的上下文和生成的答案。
"""
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

"""
节点 (application steps)
我们的程序具有2个顺序执行的steps: retrieval and generation.

我们的retrieve步骤仅仅是使用输入的问题进行相似度搜索，而generate步骤则是将检索到的上下文和原始问题格式化为聊天模型的提示，并将提示发送给llm，取得ersponse。
"""
# 定义application step，即一条Graph edge, 用于执行检索
def retrieve(state: State):
    # 通过相似度搜索，从向量数据库中取得与"question"相关的数据，通过"context"返回。
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    # 将"context"中的内容连接在一起，放在docs_content中
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # 以"question"和decs_content为参数，用prompt模型创建消息列表，保存在messages中。
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    # 以消息列表messages为参数调用llm，取得response。
    response = llm.invoke(messages)
    return {"answer": response.content}

# 将检索和生成步骤连接成一个序列。
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
# 添加从START到retrieve的edge。
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# 同步调用
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
print("\n***********************************************")

# Stream steps
for step in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="updates"
):
    print(f"{step}\n\n----------------\n")

# Stream tokens
for message, metadata in graph.stream(
    {"question": "What is Task Decomposition?"}, stream_mode="messages"
):
    print(message.content, end="|")
print("\n***********************************************")

"""
查询分析  
到目前为止，我们是使用原始输入查询来执行检索的。然而，允许模型为检索目的生成查询有一些优势。例如：  
  
除了语义搜索之外，我们还可以构建结构化的过滤器（例如，“查找2020年以来的文档”）；  
模型可以重写用户的查询，这些查询可能是多方面的或包含不相关的语言，从而生成更有效的搜索查询。  
  
查询分析利用模型从原始用户输入中转换或构建优化的搜索查询。我们可以轻松地将查询分析步骤整合到我们的应用程序中。为了说明这一点，让我们向向量存储中的文档添加一些元数据。我们将添加一些（人为的）部分到文档中，以便日后可以根据这些部分进行过滤。  
"""
# 将所有分割得到的文档块分成3部分，为3个部分文档块分别添加metadata "beginning", "middle", "end"
total_documents = len(all_splits)
third = total_documents // 3
for i, document in enumerate(all_splits):
    if i < third:
        document.metadata["section"] = "beginning"
    elif i < 2 * third:
        document.metadata["section"] = "middle"
    else:
        document.metadata["section"] = "end"

# 创建一个新的向量数据库，保存添加了metadata信息的文档块。
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(all_splits)

"""
接下来，让我们为我们的搜索查询定义一个模式。我们将使用Structured outputs来实现这一目的。这里我们定义一个查询包含一个字符串查询和一个文档部分（可以是“beginning”、“middle”或“end”），但你可以根据需要自行定义。
"""
from typing import Literal
from typing_extensions import Annotated

class Search(TypedDict):
    """Search query."""
    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# 最后，我们在LangGraph应用程序中添加一个叫做analyze_query的step，以便从用户的原始输入生成查询：
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str

# analyze_query利用LLM，分析出用户原始问题"question"中包含的"query“和”section"两部分，组成Search类型变量，做为下一个step retrieve将使用的query。
def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}

# retrieve利用vector_store的filter功能，筛选出metadata为query["section"]的文档，仅对这些筛选出的文档进行查询，这样就提高了效率。查询结果放在context中。
def retrieve(state: State):
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}

# 将context和question传递给LLM，得到response
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# 将analyze_query, retrieve, generate三个step连接成一个序列
graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
graph_builder.add_edge(START, "analyze_query")
graph = graph_builder.compile()

# 我们可以通过专门要求提供帖子末尾的上下文来测试我们的实现。请注意，模型在其回答中包含了不同的信息。
for step in graph.stream(
    {"question": "What does the end of the post say about Task Decomposition?"},
    stream_mode="updates",
):
    print(f"{step}\n\n----------------\n")