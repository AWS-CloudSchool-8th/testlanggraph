# supervisor/model/claude.py

from langchain_aws.chat_models import ChatBedrock

bedrock_claude = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1",
    credentials_profile_name=None,
    model_kwargs={
        "temperature": 0.1,
        "top_p": 0.9,
        "stop_sequences": ["\n\nHuman:"],  # Claude의 문맥 구분 트리거
    }
)

