from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages, BaseMessage


class AgentState(TypedDict):
    language: str  # Language of the quiz
    quiz_stages: int  # Number of stages of the quiz
    quiz_length: int  # Number of questions per stage
    quiz_difficulty: int  # Difficulty of the quiz
    current_stage: int  # Current stage of the quiz
    current_question: int  # Current question of the quiz
    score: int  # Current score of the player
    quiz_topic: str  # Topic of the quiz
    quiz_questions: list  # List of questions
    last_user_input: str  # Last user input
    minimum_score: float  # Minimum percentage to pass the quiz
    presenter_done: bool = False  # Whether the presenter is done
    presenter_started: bool = False  # Whether the presenter has started
    last_question: str  # Last question asked in the quiz
    last_answer: str  # Last answer given by the user
    presenter_will_speak: bool = False  # Whether the presenter will speak
    quiz_presenter_will_speak: bool = False  # Whether the quiz presenter will speak
    mocking_level: int = 0  # How much the quiz presenter will mock the user
    public_effects: bool = False  # Whether the public will react to the answer
    messages: Annotated[List[BaseMessage], add_messages]
