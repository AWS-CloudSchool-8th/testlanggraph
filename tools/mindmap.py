from langchain.tools import tool
from langchain_aws import ChatBedrock

claude = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)

@tool
def mindmap_tool(text: str) -> str:
    """
    마인드맵 요약: 중심 주제를 기준으로 분기 구조로 요약해줘.
    """
    prompt = f"""
다음 내용을 마인드맵 형식으로 요약해줘. 중심 개념에서 가지를 뻗는 구조로 정리해줘:\n{text}
가능하면 markdown 형태로 표현해줘.
"""
    response = claude.invoke(prompt)
    return response.content
mindmap_tool.name = "mindmap"
