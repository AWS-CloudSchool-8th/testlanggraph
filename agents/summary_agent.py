from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence

from tools.text_summary import text_summary_tool
from tools.table_summary import table_summary_tool
from tools.image_generation import image_generation_tool
from tools.mindmap import mindmap_tool

# Claude용 System + Human Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 다양한 요약 도구(text, table, image, mindmap)를 사용할 수 있는 AI 비서야. 사용자의 요청에 따라 적절한 도구를 선택해서 요약해."),
    ("human", "{input}")
])

# Claude LLM 선언
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)

# 사용할 Tool 리스트
tools = [
    text_summary_tool,
    table_summary_tool,
    image_generation_tool,
    mindmap_tool
]

# Claude에 tool 사용 방법을 넣기 위한 Runnable 구성
def get_agent_executor():
    tool_map = {tool.name: tool for tool in tools}

    def call_tool_and_return_result(inputs):
        tool_name = inputs["tool"]
        chunk = inputs["input"]
        if tool_name in tool_map:
            return tool_map[tool_name].invoke({"input": chunk})
        else:
            return f"[ERROR] Unknown tool: {tool_name}"

    return RunnableSequence(
        prompt,
        llm,
        lambda x: {
            "tool": x.content.strip().lower(),
            "input": x.additional_kwargs.get("input", "")
        },
        RunnableLambda(call_tool_and_return_result)
    )

