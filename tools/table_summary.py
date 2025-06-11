from langchain_core.tools import tool
from langchain_aws import ChatBedrock

claude = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1"
)

@tool
def table_summary_tool(text: str) -> str:
    """
    표 요약: 내용을 항목별로 구분해서 테이블 형태로 정리해줘.
    """
    prompt = f"다음을 표 형식으로 요약해줘. 마크다운 형태로 출력해줘:\n{text}"
    response = claude.invoke(prompt)
    return response.content

