import re
from typing import Tuple

SMALL_TALK_PATTERNS = {
    r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)[\s!.]*$": 
        "Hello! I am your document assistant. Ask me anything about your uploaded PDF!",
    r"^(thanks|thank you|thx|cheers)[\s!.]*$": 
        "You're welcome! Let me know if you have any other questions about the document.",
    r"^(bye|goodbye|see ya)[\s!.]*$": 
        "Goodbye! Have a great day ahead."
}

class IntentRouter:
    @staticmethod
    def is_small_talk(user_query: str) -> Tuple[bool, str]:
        """
        Checks if the query matches common small talk patterns.
        Returns: (is_small_talk: bool, fast_response: str)
        """
        cleaned_query = user_query.strip().lower()
        
        for pattern, response in SMALL_TALK_PATTERNS.items():
            if re.match(pattern, cleaned_query):
                return True, response

        return False, ""