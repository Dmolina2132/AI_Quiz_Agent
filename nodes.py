from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langgraph.types import interrupt
from langchain_openai import ChatOpenAI
from openai import OpenAI
from state import AgentState
from utils import (
    delayed_music,
    play_sound,
    translate,
    quiz_speak,
    LANGUAGE_CODES,
    play_background_music,
    stop_background_music,
)
from dotenv import load_dotenv
import json
from tools import speak_openai, public_reaction

load_dotenv()

presenter_tools = [speak_openai]
TOOL_DICT = {tool.name: tool for tool in presenter_tools}
model_presenter = ChatOpenAI(model_name="gpt-4o", temperature=0.5)  # .bind_tools(
#     presenter_tools
# )
model_quiz = ChatOpenAI(model_name="gpt-4o", temperature=1)
model_quiz_presenter = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)
model_final_presenter = ChatOpenAI(model_name="gpt-4.1-nano", temperature=1.5)
model_checker = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
model_public = ChatOpenAI(model_name="gpt-4o", temperature=0).bind_tools(
    tools=[public_reaction]
)


def initial_state_check(state: AgentState) -> AgentState:
    """Force the state to have the correct initial values"""
    state["language"] = state.get("language") or "en"
    state["quiz_stages"] = state.get("quiz_stages") or 2
    state["quiz_length"] = state.get("quiz_length") or 3
    state["quiz_difficulty"] = state.get("quiz_difficulty") or 6
    state["quiz_topic"] = state.get("quiz_topic") or "General Culture"
    state["minimum_score"] = state.get("minimum_score") or 0.5
    state["presenter_will_speak"] = bool(state.get("presenter_will_speak", False))
    state["quiz_presenter_will_speak"] = bool(
        state.get("quiz_presenter_will_speak", False)
    )

    # These always start with fixed values
    state["score"] = 0
    state["current_stage"] = 1
    state["current_question"] = 1
    state["messages"] = []

    return state


def show_presenter(state: AgentState):
    """Presenter introduces the quiz and talks with the user"""
    print("Running presenter node...")
    messages = state.get("messages", [])
    play_music = False
    # First time: create system prompt if there are no messages yet
    if not messages:
        play_music = True
        system_prompt = SystemMessage(
            content=f"""You are a charismatic quiz presenter with a live audience.
            The quiz is about {state["quiz_topic"]}, has {state["quiz_stages"]} stages and each stage has {state["quiz_length"]} questions.
            The minimum score to pass is {state["minimum_score"] * 100}%. All questions are multiple choice worth 1 point. The difficult of the test will be {state["quiz_difficulty"]} out of ten.
            Explain the rules, chat briefly with the user, and finally ask for their name. When you think the user is ready to start the quiz, include the token <END_INTRO> in the text.
            
            IMPORTANT:
            - You will never add <END_INTRO>, you should always wait for the first user answer, and start when he shows that he is ready to start.
            - You don't ask the questions of the quiz, you just introduce the quiz and talk with the user.
            - The language of the whole quiz will be {LANGUAGE_CODES.get(state["language"], "English")}
            - You will have a background music while you are presenting, please show enthusiasm and energy while you are presenting.
            """
        )
        messages = [system_prompt]

    # Add user input if available
    if "last_user_input" in state and state["last_user_input"]:
        messages.append(HumanMessage(content=state["last_user_input"]))
        # state["last_user_input"] = ""

    # Generate presenter response
    response = model_presenter.invoke(messages)

    # Initialize text variable
    text = response.content or ""
    presenter_done = False
    if play_music:
        delayed_music()
        play_music = False
    if state["presenter_will_speak"] and text:
        quiz_speak(text, state["language"])

    # Process tool calls (voice)
    if response.tool_calls:
        for t in response.tool_calls:
            tool_text = t["args"].get("text", "")
            # Check for <END_INTRO> in the tool call
            if "<END_INTRO>" in tool_text:
                presenter_done = True
                # Remove token before saving
                tool_text = tool_text.replace("<END_INTRO>", "")

            TOOL_DICT[t["name"]](tool_text)

            # Concatenate tool text to transcript for logging
            if tool_text and tool_text not in text:
                text = (text + " " + tool_text).strip()
    else:
        if "<END_INTRO>" in text:
            print(text)
            presenter_done = True
            # Remove token before saving
            text = text.replace("<END_INTRO>", "")
    # Append AI response to messages
    messages.append(AIMessage(content=text, type="ai"))
    state["messages"] = messages

    state["presenter_done"] = presenter_done
    if presenter_done:
        stop_background_music()

    return state


def human_node(state: AgentState) -> AgentState:
    """Node to handle human input"""
    print("Running human node...")
    value = interrupt({"question": "AI question"})
    state["last_user_input"] = value
    print("Human input: ", value)
    return state


def presenter_condition(state: AgentState) -> str:
    """Check if the presenter is done."""
    print("Running presenter condition...")
    print(state["presenter_done"])
    if state["presenter_done"]:
        print("\n🎤 End of the introduction! Moving to quiz...")
        return "quiz"
    return "human"


def quiz_maker(state: AgentState) -> AgentState:
    """Quiz maker"""
    print("Running quiz maker...")
    system_message = SystemMessage(
        content=f"""You are a world-class expert in {state["quiz_topic"]} and a quiz maker in a TV show. The presenter has already introduced the quiz and talked with the user.
    Your duty is to create questions for the quiz. The quiz is about {state["quiz_topic"]}, has {state["quiz_stages"]} stages and each stage has {state["quiz_length"]} questions. 
    The difficulty of the quiz is {state["quiz_difficulty"]} out of ten. 
    - 1-3: basic factual questions
    - 4-6: questions that require some reasoning or understanding
    - 7-10: questions that require multi-step reasoning, applying concepts, or analyzing scenarios
    The language of the quiz will be {LANGUAGE_CODES.get(state["language"], "English")}
    Make sure that you creat questions adapted to the difficulty.
    For each stage, you will have to think about one subtopic of that topic and create questions about it.
    The questions will be multiple choice questions with 4 options each. The questions will be stored in a dictionary. To each questions you will assign a letter that will be used to
    answer the question. The correct answer must be stored in the dictionary with the key "correct_answer". 
    Example of the output:"""
        + """
    {"1-1":{"question": "...", "options": [["a", "..."], ["b", "..."], ["c", "..."], ["d", "..."]], "correct_answer": "a"}}
    Where the key 1-1 means stage 1, question 1. Please only return the dictionary, no other text.
    """
    )
    response = model_quiz.invoke([system_message])

    # try:
    #     # Parse text
    #     questions_dict = json.loads(response.content)
    # except json.JSONDecodeError:
    #     print("⚠️ Warning: JSON parsing failed. Output was:")
    #     print(response.content)
    #     questions_dict = {}

    # Save state
    state["quiz_questions"] = response.content

    return state


def check_json(state: AgentState) -> AgentState:
    """Check if the json is valid and rewrite it with an agent if not"""
    try:
        json.loads(state["quiz_questions"])
    except json.JSONDecodeError as e:
        system_message = SystemMessage(
            content=f"""You are a json reviewer and your duty is to correct if the following json is valid. You have the following json:
        {state["quiz_questions"]}
        and the error is:
        {e}
        Please correct the json and return it. Return Only the json without any other message, it must be ready to be converted to a dictionary.
        """
        )
        response = model_checker.invoke([system_message])
        state["quiz_questions"] = json.loads(response.content)
    return state


def question_node(state: AgentState) -> AgentState:
    """Here we will ask the questions to the user"""
    question = state["quiz_questions"][
        str(state["current_stage"]) + "-" + str(state["current_question"])
    ]
    formatted_questions = "\n".join(
        [f"{option[0]}. {option[1]}" for option in question["options"]]
    )
    question_text = f"""Here we have the question {state["current_question"]} of stage {state["current_stage"]}: \n {question["question"]}\n {formatted_questions}"""

    question_message = AIMessage(
        content=translate(question_text, state["language"]), type="ai"
    )
    return {**state, "messages": [question_message]}


def human_node_quiz(state: AgentState) -> AgentState:
    """Node to handle human input"""
    value = interrupt({"question": "What is your answer?"})
    state["last_user_input"] = value
    return state


def evaluate_answer(state: AgentState) -> AgentState:
    """Evaluate the answer of the user"""
    state["last_question"] = state["quiz_questions"][
        str(state["current_stage"]) + "-" + str(state["current_question"])
    ]
    state["last_answer"] = state["last_user_input"]
    if (
        state["last_user_input"]
        == state["quiz_questions"][
            str(state["current_stage"]) + "-" + str(state["current_question"])
        ]["correct_answer"]
    ):
        # Use prints instead of ai messages
        play_sound("correct")
        print(
            f"🟢🟢🟢🟢🟢🟢🟢🟢{translate('Correct!', state['language'])}🟢🟢🟢🟢🟢🟢🟢🟢🟢"
        )
        state["score"] += 1
    else:
        # Use prints instead of ai messages
        play_sound("wrong")
        print(
            f"🔴🔴🔴🔴🔴🔴🔴{translate('Wrong!', state['language'])} {translate('The correct answer was', state['language'])} {state['quiz_questions'][str(state['current_stage']) + '-' + str(state['current_question'])]['correct_answer']} 🔴🔴🔴🔴🔴🔴🔴"
        )
    state["current_question"] += 1
    if state["current_question"] > state["quiz_length"]:
        state["current_question"] = 1
        state["current_stage"] += 1
    return state


def quiz_public(state: AgentState) -> AgentState:
    """Simulate the public in the quiz show"""
    if state["public_effects"]:
        input_message = f"""You are the public in a quiz show. Your job is to cheer, mock or boo the user.
        The question you have to react to is {state["last_question"]["question"]},
        You have a tool that allows you to select the reaction you want to make and play the sound, please use it to reflect your reaction to the answer of the user. This tool
        allows you to react suprised, cheer, boo, laugh or be suprised.
        For instance, if the user get the correct answer you can cheer him, but if the user get the wrong answer you can boo him, be suprised if the answer is close to 
        the correct answer or laugh if the answer is wrong and the question was very easy.
        the options were {state["last_question"]["options"]}, the correct answer is {state["last_question"]["correct_answer"]}
        and the answer of the user was {state["last_answer"]}.
        
        You must always call the tool, never return a text answer."""
        response = model_public.invoke([SystemMessage(content=input_message)])
        state["messages"].append(response)
        return state
    else:
        return state


def should_use_public_tool(state: AgentState) -> bool:
    """Check if the public should react to the answer of the user"""
    if state["public_effects"]:
        return "public"
    else:
        return "continue"


def quiz_commenter(state: AgentState) -> AgentState:
    """Will do comments based on the answer of the user"""

    input_message = f"""You are a charismatic presenter who is commenting on a quiz. Your job is only to bring joy to the show, with
    funny and ingenious comments about the answers and the questions, you can be a bit mocker as well. The question you have to comment is {state["last_question"]["question"]},
    the options were {state["last_question"]["options"]}, the correct answer is {state["last_question"]["correct_answer"]}
    and the user answer was {state["last_answer"]}.
    Answer only with a couple of sentences, don't extend too much.

    MOCKING LEVEL: {state["mocking_level"]}. Mocking level describes how much the presenter will mock the user.
    0: No mocking
    1: Light mocking
    2: Moderate mocking
    3: Heavy mocking
    4: Extreme mocking
    5: Mocking like a comedian, you can even be a bit cruel with the user when he fails.
    NOTES: The language of the quiz will be {LANGUAGE_CODES.get(state["language"], "English")}"""
    response = model_quiz_presenter.invoke([SystemMessage(content=input_message)])
    if state["quiz_presenter_will_speak"]:
        quiz_speak(response.content, state["language"])
    state["messages"].append(AIMessage(content=response.content, type="ai"))
    return state


def should_continue(state: AgentState) -> bool:
    """Check if the quiz have finished and if the user has passed, this will happen when current_stage > quiz_stages or score >= minimum_score"""
    if (
        state["current_stage"] > state["quiz_stages"]
        #     or (
        #     state["score"]
        #     >= state["minimum_score"] * state["quiz_length"] * state["quiz_stages"]
        # )
    ):
        return "end"

    return "continue"


def final_comment(state: AgentState) -> AgentState:
    """Will do a final comment based on the score of the user"""
    input_message = f"""You are a charismatic presenter who is commenting the end of the quiz. The user has finished the quiz and his score is {state["score"]}.
    Then minimum score to pass is {state["minimum_score"] * state["quiz_length"] * state["quiz_stages"]}. The quizz was about {state["quiz_topic"]}.
    Close the program with some comments on the result. 
    NOTES: The language of the quiz was {LANGUAGE_CODES.get(state["language"], "English")}
    """
    response = model_final_presenter.invoke([SystemMessage(content=input_message)])
    if (
        state["score"]
        >= state["minimum_score"] * state["quiz_length"] * state["quiz_stages"]
    ):
        play_sound("final_win")
    else:
        play_sound("final_lose")
    if state["presenter_will_speak"]:
        quiz_speak(response.content, state["language"])

    state["messages"].append(AIMessage(content=response.content, type="ai"))
    return state
