import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from backend.config import settings
from backend.utils.logger import logger

# NOTE: Literal JSON braces are doubled {{ }} because ChatPromptTemplate treats
# single { } as template variables (only {text} is a real variable here).
EXTRACTION_PROMPT = """You are an expert Knowledge Graph builder.
Extract factual (subject, relation, object) triples from the text snippet below.

Rules:
1. Extract EVERY clear fact: what a component, service, system or product IS, USES, STORES,
   MANAGES, PROVIDES, HANDLES, INTEGRATES_WITH, SENDS, DEPENDS_ON, HAS_TABLE, etc.
2. If the snippet starts with a [Section: ...] label or a heading, use that section's main name
   (for example "Order Service") as the subject of its bullet points.
3. Keep entity names short and canonical (e.g. "PostgreSQL", "Redis", "Order Service").
4. Relations must be UPPER_SNAKE_CASE verbs (e.g. USES, STORES, MANAGES, HAS_TABLE).
5. Return an empty array [] ONLY if the text has no facts at all (e.g. it is just a title).
6. Output a valid JSON array ONLY, with no explanation, in this exact format:
[
  {{"subject": "Order Service", "relation": "STORES_DATA_IN", "object": "PostgreSQL"}},
  {{"subject": "Order Service", "relation": "HAS_TABLE", "object": "orders"}}
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

    @traceable(name="kg_extract_triples", run_type="chain")
    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        # Short one-line preview of the chunk, used only in log messages
        preview = text[:70].replace("\n", " ")
        content = ""
        try:
            messages = self.prompt.format_messages(text=text)
            response = self.llm.invoke(messages)
            content = response.content.strip()

            # Clean up JSON backticks if present
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            triples = json.loads(content)

            # Some LLM replies wrap the list, e.g. {"triples": [...]} -> take the first list inside
            if isinstance(triples, dict):
                inner = [v for v in triples.values() if isinstance(v, list)]
                logger.warning(f"[KG DEBUG] LLM returned a JSON object, not a list. Keys: {list(triples.keys())} | chunk: {preview!r}")
                triples = inner[0] if inner else []

            if not isinstance(triples, list):
                logger.warning(f"[KG DEBUG] Unexpected JSON type {type(triples).__name__} | raw: {content[:200]!r} | chunk: {preview!r}")
                return []

            if not triples:
                logger.warning(f"[KG DEBUG] 0 triples | raw LLM output: {content[:200]!r} | chunk: {preview!r}")
            else:
                logger.info(f"[KG DEBUG] {len(triples)} triples | chunk: {preview!r}")
            return triples

        except Exception as e:
            logger.warning(f"Failed to extract KG triples for chunk: {str(e)} | raw: {content[:200]!r} | chunk: {preview!r}")
            return []