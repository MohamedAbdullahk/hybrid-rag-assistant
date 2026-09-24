import re
from typing import Tuple
from backend.utils.logger import logger

# Prompt Injection and Jailbreak patterns
INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"system prompt",
    r"you are now an? (unrestricted|evil|dan)",
    r"bypass (the )?filter",
    r"override your rules",
    r"jailbreak"
]

# Sensitive patterns that should never leak in outputs
SENSITIVE_OUTPUT_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",  # OpenAI API key pattern
    r"DATABASE_URL",
    r"SECRET_KEY",
    r"OPENAI_API_KEY"
]

class GuardrailsManager:
    @staticmethod
    def validate_input(user_query: str) -> Tuple[bool, str]:
        """
        Validates input query against security guidelines.
        Returns: (is_safe: bool, warning_message: str)
        """
        cleaned_query = user_query.strip().lower()

        # Check for Prompt Injection / Jailbreaks
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, cleaned_query):
                logger.warning(f"Guardrails blocked input matching pattern: {pattern}")
                return False, "Your query contains unauthorized instructions and was blocked."

        return True, ""

    @staticmethod
    def sanitize_output(response_text: str) -> str:
        """
        Sanitizes output response to prevent information leakage.
        """
        sanitized = response_text

        # Redact API keys or sensitive system parameters if present
        for pattern in SENSITIVE_OUTPUT_PATTERNS:
            sanitized = re.sub(pattern, "[REDACTED_SENSITIVE_INFO]", sanitized)

        return sanitized

guardrails = GuardrailsManager()