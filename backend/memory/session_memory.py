from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class SessionMemoryManager:
    """
    Manages per-session conversation memory.
    Maintains a sliding window of the last 6 turns (12 messages total).
    """
    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        # Dictionary storing chat messages per session_id
        self.sessions: Dict[str, List[BaseMessage]] = {}

    def get_history(self, session_id: str) -> List[BaseMessage]:
        """Returns the list of raw BaseMessage objects for a given session."""
        return self.sessions.get(session_id, [])

    def add_turn(self, session_id: str, user_query: str, ai_response: str):
        """Adds a new question-answer turn to the session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(HumanMessage(content=user_query))
        self.sessions[session_id].append(AIMessage(content=ai_response))

        # Sliding window: keep only the last window_size turns (2 messages per turn)
        max_messages = self.window_size * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def get_formatted_history(self, session_id: str) -> str:
        """Formats session history into a readable string for LLM context."""
        history = self.get_history(session_id)
        if not history:
            return "No previous history."

        formatted_str = ""
        for msg in history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted_str += f"{role}: {msg.content}\n"
        return formatted_str.strip()

# Global singleton instance for local runtime
memory_manager = SessionMemoryManager()