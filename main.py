from graph import build_graph
from state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

initial_state = AgentState(
    language="de",
    quiz_stages=2,
    quiz_length=2,
    quiz_difficulty=3,  # Difficulty out of ten
    current_stage=1,
    current_question=1,
    score=0,
    quiz_topic="Spain",
    quiz_questions={},
    last_user_input="",
    minimum_score=0.5,
    messages=[],
    presenter_started=False,
    presenter_done=False,
    quiz_presenter_will_speak=True,
    presenter_will_speak=False,
    mocking_level=5,
    public_effects=True,
)
state = initial_state

memory = InMemorySaver()
app = build_graph(checkpointer=memory)

thread_config = {"configurable": {"thread_id": "quiz-thread-1"}}


# Process output in the stream
def run_stream(input_state, last_answer={"answer": ""}):
    for event in app.stream(input_state, config=thread_config):
        last_event_value = list(event.values())[-1]
        last_event_key = list(event.keys())[-1]

        if (last_event_key != "__interrupt__") and ("human" not in last_event_key):
            if ("messages" in last_event_value) and (
                last_event_value["messages"] != []
            ):
                if isinstance(last_event_value["messages"][-1], AIMessage):
                    if (
                        last_answer["answer"]
                        == last_event_value["messages"][-1].content
                    ):
                        continue
                    last_answer["answer"] = last_event_value["messages"][-1].content
                    print(f"\n🎤 {last_event_value['messages'][-1].content}")
        elif last_event_key == "__interrupt__":
            return event
    return None


# First execution in stream
interrupt = run_stream(state)

while True:
    if interrupt:
        # User input
        print("Waiting for user input...")
        user_input = input("👤: ")

        # Resume from interrupt
        command = Command(resume=user_input)
        interrupt = run_stream(command)
    else:
        # Check if finished
        current_thread_state = app.get_state(config=thread_config)
        if not getattr(current_thread_state, "next"):
            print("\n🏆 End of the quiz!")
            break
