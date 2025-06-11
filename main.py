from graph.graph_builder import run_pipeline
from utils.caption_loader import load_caption_from_s3

def main():
    # ▶️ 여기에 S3 버킷 이름과 caption.txt 위치 설정
    bucket_name = "caption.txt"
    key = "captions/s9igkeyj5QI.txt"

    caption = load_caption_from_s3(bucket_name, key)
    if not caption:
        print(" 자막 불러오기 실패. 종료합니다.")
        return

    graph = run_pipeline(caption)
    result = graph.invoke({"caption": caption})

    print("\n 최종 요약 결과:\n")
    for tool, output in result["results"]:
        print(f"\n [{tool}] →\n{output}")

if __name__ == "__main__":
    main()

