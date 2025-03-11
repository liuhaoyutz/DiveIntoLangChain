from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

#使用本地服务创建模型
model = ChatOpenAI(
    #model_name="deepseek-r1:32b",
    model_name="qwen2",
    openai_api_base="http://127.0.0.1:11434/v1",
    openai_api_key="EMPTY",
    streaming=False
)

#response = model.invoke([HumanMessage(content="你帮我算下，3.941592623412424+4.3434532535353的结果")])
#print(response.content)

# 下面开始使用ReAct机制，定义工具，让LLM使用工具做专业的事情。

# 定义工具，要继承自LangChain的BaseTool
class SumNumberTool(BaseTool):
    name: str = "加法计算工具"
    description: str = "当你被要求计算2个数字相加时，使用此工具"

    def _run(self, a, b):
        # 检查输入是否为字典并尝试提取值
        if isinstance(a, dict) and 'title' in a:
            try:
                a = float(a.get('value', 0))  # 假设数值存储在'value'键下
            except ValueError:
                return "无法将输入A转换为数字，请检查输入格式。"
        if isinstance(b, dict) and 'title' in b:
            try:
                b = float(b.get('value', 0))  # 假设数值存储在'value'键下
            except ValueError:
                return "无法将输入B转换为数字，请检查输入格式。"

        # 将输入转换为float类型
        try:
            a = float(a)
            b = float(b)
        except ValueError:
            return "输入包含非数值字符，请输入有效的数字。"
        
        return a + b
        
# 工具合集
tools = [SumNumberTool()]

# 提示词，直接从langchain hub上下载，因为写这个ReAct机制的prompt比较复杂，直接用现成的。
prompt = hub.pull("hwchase17/structured-chat-agent")

# 定义AI Agent
agent = create_structured_chat_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

# 使用Memory记录上下文
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)

# 定义AgentExecutor，必须使用AgentExecutor，才能执行代理定义的工具
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True
)

# 测试使用到工具的场景
agent_executor.invoke({"input": "你帮我算下3.941592623412424+4.3434532535353的结果"})

# 测试不使用工具的场景
agent_executor.invoke({"input": "请你充当稿件审核师，帮我看看'''号里的内容有没有错别字，如果有的话帮我纠正下。'''今天班级里的学生和老实要去哪里玩'''"})        