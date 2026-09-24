import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from backend.config import settings
from backend.utils.logger import logger

EXTRACTION_PROMPT = """You are an expert Knowledge Graph builder.
Extract key entity-relationship triples from the text snippet below.

Rules:
1. Extract relationships as (subject, relation, object).
2. Keep entities simple and canonical (e.g., "Company A", "2022", "$2.1B").
3. Output valid JSON array ONLY, in this exact format:
[
  {"subject": "Company A", "relation": "ACQUIRED", "object": "Company B"},
  {"subject": "Company A", "relation": "PAID_AMOUNT", "object": "$2.1 Billion"}
]

Text snippet:
{text}
"""

class KGExtractor:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)

    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        try:
            formatted_prompt = self.prompt.format(text=text)
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Clean up JSON backticks if present
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            triples = json.loads(content)
            if isinstance(triples, list):
                return triples
            return []
        except Exception as e:
            logger.warning(f"Failed to extract KG triples for chunk: {str(e)}")
            return []