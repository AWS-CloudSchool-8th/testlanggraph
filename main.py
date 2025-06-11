from graph.graph_builder import run_pipeline
from utils.caption_loader import load_caption_from_file

def main():
    caption = load_caption_from_file()
    graph = run_pipeline(caption)
    result = graph.invoke({"caption": caption})
    print("\n✅ 최종 요약 결과:\n")
    for tool, output in result["results"]:
        print(f"\n📌 [{tool}] → {output}")

if __name__ == "__main__":
    main()
