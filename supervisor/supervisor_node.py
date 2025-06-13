
# supervisor/supervisor_node.py

import json
from supervisor.model.claude import bedrock_claude
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("""
다음은 유튜브 영상의 자막입니다:

{caption}

이 자막을 **내용 흐름**이나 **주제**별로 나누고, 각 부분마다 아래 도구 중 어떤 요약 방식이 적절할지 판단해줘.  
**절대 요약하지 말고**, **도구 판단만 해줘**.  
각 의미단위의 **시작 시간, 끝 시간**을 추정해서 다음 JSON 형식으로 출력해줘줘:
[
  {{
    "chunk": "[at 0.00 seconds] ~ [at 77.00 seconds] 내용",
    "tools": ["text", "table"]
  }},
  ...
]


도구 설명:
- "text": 일반적인 설명이나 줄글로 표현하기 좋은 경우
- "table": 분류, 비교, 나열처럼 구조화된 정보
- "mindmap": 중심 개념에서 가지를 뻗는 형식이 적절할 때
- "image": 시각적으로 보여주면 좋을 설명일 때

이 외에는 아무 말도 하지 마!  
형식은 반드시 JSON 배열로만 답해줘.  
절대 요약하지 마! 예시도 들지 마!                                               
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



