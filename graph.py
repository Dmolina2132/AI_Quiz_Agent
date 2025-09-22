from langgraph.graph import StateGraph, START, END

# import langgraph tool nodes
from langgraph.prebuilt import ToolNode
from state import AgentState
from tools import public_reaction
import os
import nodes


def build_graph(checkpointer=None, save_image=False, image_path="graph.png"):
    graph = StateGraph(AgentState)
    graph.add_node("initial_state_check", nodes.initial_state_check)
    graph.add_node("presenter", nodes.show_presenter)
    graph.add_node("human", nodes.human_node)
    graph.add_node("human_quiz", nodes.human_node_quiz)
    # tool_node_presenter = ToolNode(tools.speak)
    # graph.add_node("tool_node_presenter", tool_node_presenter)
    graph.add_node("quiz", nodes.quiz_maker)
    graph.add_node("check_json", nodes.check_json)
    graph.add_node("question", nodes.question_node)
    graph.add_node("evaluate", nodes.evaluate_answer)
    graph.add_node("public", nodes.quiz_public)
    toolnode = ToolNode(tools=[public_reaction])
    graph.add_node("tool_node_public", toolnode)
    graph.add_node("comment", nodes.quiz_commenter)
    graph.add_node("final_comment", nodes.final_comment)
    graph.add_edge(START, "initial_state_check")
    graph.add_edge("initial_state_check", "presenter")
    # graph.add_edge("presenter", "tool_node_presenter")
    # graph.add_edge("tool_node_presenter", "presenter")
    graph.add_edge("human", "presenter")
    graph.add_conditional_edges(
        "presenter",
        nodes.presenter_condition,
        {"quiz": "quiz", "human": "human"},
    )
    graph.add_edge("quiz", "check_json")
    graph.add_edge("check_json", "question")
    graph.add_edge("question", "human_quiz")
    graph.add_edge("human_quiz", "evaluate")
    graph.add_edge("evaluate", "public")
    graph.add_conditional_edges(
        "public",
        nodes.should_use_public_tool,
        {"public": "tool_node_public", "continue": "comment"},
    )
    graph.add_edge("tool_node_public", "comment")
    graph.add_conditional_edges(
        "comment",
        nodes.should_continue,
        {"end": "final_comment", "continue": "question"},
    )
    graph.add_edge("final_comment", END)

    app = graph.compile(checkpointer=checkpointer)
    if save_image:
        try:
            # Obtener la representación del grafo
            print("Generating graph image...")
            graph_image = app.get_graph().draw_mermaid_png()

            # Guardar la imagen
            print("Saving graph image...")
            with open(image_path, "wb") as f:
                f.write(graph_image)

            print(f"Imagen del grafo guardada en: {os.path.abspath(image_path)}")

        except Exception as e:
            print(f"Error al generar la imagen del grafo: {e}")
            print("Asegúrate de tener instalado: pip install pygraphviz")
    return app
