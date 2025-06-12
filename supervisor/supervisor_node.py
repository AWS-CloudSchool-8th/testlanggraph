
# supervisor/supervisor_node.py

import json
from supervisor.model.claude import bedrock_claude
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
다음은 유튜브 영상의 자막입니다:

{caption}

이 자막을 의미 단위(주제별, 내용 흐름별)로 나누고, 각 의미 덩어리에 대해 아래 중 어떤 요약 방식이 가장 적절할지 판단해줘(복수 선택 가능).
요약 방식만 판단해주고, 절대 너가 요약하지마. 
- text: 줄글 요약이 적절한 경우
- table: 의미 덩어리가  구조화된 비교·분류·나열일 경우
- mindmap: 마인드맵으로 시각화하면 좋을 경우
- image: 시각적으로 나타낼 수 있는 설명
- audio: 배경음악이나 음향 효과 관련 내용인 경우

각 의미단위의 **시작 시간, 끝 시간**을 추정해서 다음 JSON 형식으로 출력해줘:
[
  {{
    "chunk": "[at 0.00 seconds] ~ [at 77.00 seconds] 내용",
    "tools": ["text", "table"]
  }},
  ...
]
""")

def extract_json_from_response(response: str):
    import re

    try:
        json_start = response.find("[")
        json_end = response.rfind("]") + 1
        json_part = response[json_start:json_end]

        # JSON 객체들만 추출
        object_matches = re.finditer(r'{.*?}', json_part, re.DOTALL)
        items = []
        for match in object_matches:
            try:
                obj = json.loads(match.group())
                items.append(obj)
            except json.JSONDecodeError:
                continue  # 잘린 항목은 무시
        return items

    except Exception as e:
        print("JSON 수정 실패:", e)
        print("JSON 문자열:\n", response)
        raise


def analyze_caption(caption: str):
    print(" Supervisor: 자림 분석 중...")
    chain = prompt_template | bedrock_claude
    response = chain.invoke({"caption": caption})
    print("==== Claude 응답 원문 ====")
    print(response.content)
    print("==== ↑ 응답 끝 ====")
    return extract_json_from_response(response.content)

