# 🤖 AI Quiz with LangGraph  

An interactive **quiz game powered by LangGraph and LLMs**, where the logic of the game is modeled as a graph of states.  
The quiz dynamically generates questions, evaluates answers, provides feedback, and tracks player scores.  

This project demonstrates how to use **LangGraph** to build structured workflows on top of Large Language Models (LLMs).  

---

## ✨ Features
- **Dynamic Question Generation**: questions created by an LLM based on the selected topic.  
- **Game Flow with LangGraph**: start → question → answer → feedback → next question / end.  
- **Scoring System**: points for correct answers, limited lives for mistakes.  
- **Learning Mode**: optional mode where each answer includes a short explanation.  
- **Persistence**: saves player progress and scores to JSON (or SQLite).  
- **Extensible**: easy to add more nodes (e.g., category selection, difficulty levels).  

---

## 🛠️ Tech Stack
- [LangGraph](https://github.com/langchain-ai/langgraph)  
- [LangChain](https://www.langchain.com/)  
- [OpenAI API](https://platform.openai.com/)  
- Python 3.10+  

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-quiz-langgraph.git
