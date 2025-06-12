from langchain_core.tools import tool
from langchain_aws import ChatBedrock

claude = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)

@tool
def text_summary_tool(text: str) -> str:
    """
    줄글 요약: Claude 3에게 요약 프롬프트를 보내서 줄글 형태의 요약을 받는다.
    """
    prompt = f"다음을 간결하게 줄글로 요약해줘:\n{text}"
    response = claude.invoke(prompt)
    return response.content
text_summary_tool.name = "text"
