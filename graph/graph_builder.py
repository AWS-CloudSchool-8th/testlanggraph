from langgraph.graph import StateGraph, END
from supervisor.supervisor_node import analyze_caption
from agents.summary_agent import get_agent_executor

def run_pipeline(caption: str):
    graph = StateGraph(dict)  # 상태는 딕셔너리

    # Supervisor Node - 의미 단위로 자막 나누고, 요약 툴 판단
    def supervisor_node(state):
        print(" Supervisor: 자막 분석 중...")
        chunks = analyze_caption(state["caption"])
        state["chunks"] = chunks
        return state

    # 2️⃣ Agent Node - 각 의미 chunk마다 tool 실행
    def run_agent_node(state):
        print("Agent: 요약 도구 실행 중...")
        executor = get_agent_executor()
        results = []

        for item in state["chunks"]:
            input_text = item["chunk"]
            for tool in item["tools"]:
                print(f" [Tool: {tool}] 실행 중...")
                try:
                    output = executor.invoke({
                        "input": input_text,
                        "tool": tool
                    })
                    results.append((tool, output))
                except Exception as e:
                    print(f" Tool 실행 실패: {e}")
                    results.append((tool, f"[실패] {str(e)}"))

        state["results"] = results
        return state

    # 3️⃣ 그래프 정의
    graph.add_node("Supervisor", supervisor_node)
    graph.add_node("Agent", run_agent_node)

    graph.set_entry_point("Supervisor")
    graph.add_edge("Supervisor", "Agent")
    graph.add_edge("Agent", END)

    return graph.compile()

