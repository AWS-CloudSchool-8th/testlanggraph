from langchain_core.tools import tool
import boto3
import base64
import os
from datetime import datetime

# Bedrock 클라이언트 생성
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

@tool
def image_generation_tool(text: str) -> str:
    """
    Titan Image 모델을 이용해 실제 이미지를 생성하고 저장 경로를 반환합니다.
    """
    prompt = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": text[:100]
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "quality": "standard",
            "height": 512,
            "width": 512,
            "cfgScale": 8.0
        }
    }

    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-image-generator-v1",
            contentType="application/json",
            accept="application/json",
            body=str(prompt).replace("'", '"')  # JSON 문자열로 변환
        )

        response_body = response["body"].read()
        image_data = eval(response_body)["images"][0]  # base64 encoded
        binary_data = base64.b64decode(image_data)

        # 로컬에 저장
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"titan_image_{now}.png"
        with open(save_path, "wb") as f:
            f.write(binary_data)

        return f" Titan Image 생성 완료 → {save_path}"
    except Exception as e:
        return f" Titan Image 생성 실패: {e}"
image_generation_tool.name = "image"
