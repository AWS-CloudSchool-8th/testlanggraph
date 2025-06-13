from dotenv import load_dotenv
import os
import requests
import boto3

from graph.graph_builder import run_pipeline
from utils.caption_loader import load_caption_from_s3

from langchain_core.tracers import LangChainTracer

# 📦 .env 로드
load_dotenv()
API_KEY = os.getenv("VIDCAP_API_KEY")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
CAPTION_PREFIX = "captions/"

# ✅ LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
tracer = LangChainTracer(project_name="LangGraph-Summarizer")

# 🎯 유튜브 자막 요청 + S3 업로드
def fetch_and_upload_caption(youtube_url, api_key, bucket_name, key, locale="ko"):
    api_url = "https://vidcap.xyz/api/v1/youtube/caption"
    params = {"url": youtube_url, "locale": locale}
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"[*] 유튜브 자막 요청 중: {youtube_url}")
    try:
        response = requests.get(api_url, params=params, headers=headers)
        response.raise_for_status()
        text = response.json().get("data", {}).get("content", "")
        if not text.strip():
            print(" 자막은 응답되었으나 내용이 없습니다.")
            return False
    except requests.exceptions.RequestException as e:
        print(f" 요청 실패: {e}")
        return False

    # ▶️ S3 업로드
    s3 = boto3.client("s3")
    try:
        s3.put_object(Bucket=bucket_name, Key=key, Body=text.encode("utf-8"))
        print(f"자막 S3 업로드 완료: s3://{bucket_name}/{key}")
        return True
    except Exception as e:
        print(f"S3 업로드 실패: {e}")
        return False

# 🚀 전체 요약 실행 메인 함수
def main():
    youtube_url = input(" 유튜브 링크를 입력하세요: ").strip()
    print(f"[디버그] 입력된 URL: {youtube_url}")

    # 유튜브 video_id 추출
    video_id = youtube_url.split("v=")[-1].split("&")[0]
    s3_key = f"{CAPTION_PREFIX}{video_id}.txt"

    # 자막 요청 및 S3 업로드
    success = fetch_and_upload_caption(youtube_url, API_KEY, BUCKET_NAME, s3_key)
    if not success:
        print("자막 처리 실패. 종료합니다.")
        return

    # S3에서 자막 불러오기
    caption = load_caption_from_s3(BUCKET_NAME, s3_key)
    if not caption:
        print("자막 불러오기 실패. 종료합니다.")
        return

    # 요약에 사용된 자막 확인
    print("\n====== 실제 요약에 사용된 caption 앞부분 ======")
    print(caption[:300])
    print("=============================================")

    # LangGraph 파이프라인 실행
    graph = run_pipeline(caption)

    # ✅ LangSmith 추적 포함 실행
    result = graph.invoke(
        {"caption": caption},
        config={"callbacks": [tracer]}
    )

    print("\n최종 요약 결과:\n")
    for tool, output in result["results"]:
        print(f"\n[{tool}] →\n{output}")

if __name__ == "__main__":
    main()

