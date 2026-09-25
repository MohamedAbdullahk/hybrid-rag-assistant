import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langsmith import traceable
from backend.config import settings
from backend.services.rag_service import rag_service_instance
from backend.utils.logger import logger

# All RAG strategies to compare
STRATEGIES = ["baseline", "hybrid_rerank", "fusion", "crag"]

# Overall score >= this value is considered PASS
PASS_THRESHOLD = 0.7

JUDGE_PROMPT = """You are a strict evaluator for a RAG (Retrieval-Augmented Generation) system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Expected Answer (reference):
{expected}

Score each metric from 0.0 to 1.0:
1. faithfulness: Is every claim in the Generated Answer supported by the Retrieved Context? (1.0 = fully supported, 0.0 = made up)
2. answer_relevancy: Does the Generated Answer directly address the Question and agree with the Expected Answer? (1.0 = fully, 0.0 = not at all)
3. context_relevancy: Does the Retrieved Context contain the information needed to answer the Question? (1.0 = all needed info present, 0.0 = none)

Return ONLY valid JSON in this exact format:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_relevancy": 0.0, "reason": "one short sentence"}}
"""


class RAGEvaluator:
    def __init__(self):
        # LLM-as-a-Judge: same model family, temperature 0 for consistent scoring
        self.judge = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

    @traceable(name="llm_judge", run_type="chain")
    def _judge(self, question: str, contexts: List[str], answer: str, expected: str) -> Dict[str, Any]:
        """Asks the judge LLM to score one answer. Returns 0.0 scores on failure."""
        prompt = JUDGE_PROMPT.format(
            question=question,
            context="\n---\n".join(contexts) if contexts else "No context retrieved.",
            answer=answer,
            expected=expected or "Not provided."
        )
        try:
            content = self.judge.invoke(prompt).content.strip()

            # Clean up ```json fences if the model adds them
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            data = json.loads(content)
            return {
                "faithfulness": float(data.get("faithfulness", 0.0)),
                "answer_relevancy": float(data.get("answer_relevancy", 0.0)),
                "context_relevancy": float(data.get("context_relevancy", 0.0)),
                "reason": str(data.get("reason", ""))
            }
        except Exception as e:
            logger.warning(f"Judge LLM failed to score: {str(e)}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_relevancy": 0.0,
                "reason": f"Judge failed: {str(e)}"
            }

    @traceable(name="evaluate_strategy", run_type="chain")
    def evaluate_query(
        self,
        session_id: str,
        query: str,
        expected_output: str,
        strategy: str = "hybrid_rerank"
    ) -> Dict[str, Any]:
        """Runs ONE strategy end-to-end and scores its answer."""
        logger.info(f"Running Evaluation for Strategy: '{strategy}' on query: '{query}'")

        # 1. Generate answer with the SELECTED strategy (retrieval happens only once)
        result = rag_service_instance.answer_question(
            session_id, query, strategy=strategy, save_memory=False
        )
        contexts = result.get("contexts", [])
        actual_output = result["answer"]

        # 2. Judge the answer
        scores = self._judge(query, contexts, actual_output, expected_output)
        overall = round(
            (scores["faithfulness"] + scores["answer_relevancy"] + scores["context_relevancy"]) / 3, 3
        )

        summary = {
            "strategy": strategy,
            "faithfulness_score": round(scores["faithfulness"], 3),
            "relevancy_score": round(scores["answer_relevancy"], 3),
            "context_relevancy_score": round(scores["context_relevancy"], 3),
            "overall_score": overall,
            "passed": overall >= PASS_THRESHOLD,
            "latency_seconds": result.get("timings", {}).get("total", 0.0),
            "reason": scores["reason"],
            "actual_output": actual_output
        }

        logger.info(f"Evaluation Results ({strategy}): {summary}")
        return summary

    @traceable(name="compare_all_strategies", run_type="chain", tags=["evaluation"])
    def compare_all_strategies(
        self,
        session_id: str,
        query: str,
        expected_output: str
    ) -> Dict[str, Any]:
        """Runs all strategies on the same question and picks the best one."""
        results = [
            self.evaluate_query(session_id, query, expected_output, strategy=s)
            for s in STRATEGIES
        ]

        # Best = highest overall score; if tied, the faster one wins
        best = max(results, key=lambda r: (r["overall_score"], -r["latency_seconds"]))

        return {
            "query": query,
            "threshold": PASS_THRESHOLD,
            "best_strategy": best["strategy"],
            "results": results
        }


evaluator_instance = RAGEvaluator()