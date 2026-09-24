import os
from typing import List, Dict, Any
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from backend.services.rag_service import rag_service_instance
from backend.rag.strategies import StrategyEngine
from backend.utils.logger import logger

class DeepEvalEvaluator:
    def __init__(self):
        # Initialize DeepEval Metrics with default parameters for compatibility
        self.faithfulness = FaithfulnessMetric()
        self.relevancy = AnswerRelevancyMetric()
        
        # Explicitly set threshold properties
        self.faithfulness.threshold = 0.7
        self.relevancy.threshold = 0.7

    def evaluate_query(
        self, 
        session_id: str, 
        query: str, 
        expected_output: str, 
        strategy: str = "hybrid_rerank"
    ) -> Dict[str, Any]:
        """
        Runs a test query through a specific RAG strategy and evaluates output using DeepEval.
        """
        logger.info(f"Running DeepEval for Strategy: '{strategy}' on query: '{query}'")

        # 1. Retrieve Context and Generate Answer for strategy
        strategy_engine = StrategyEngine()
        retrieved_data = strategy_engine.retrieve_context(session_id, query, strategy=strategy)
        
        retrieved_contexts = [doc.page_content for doc in retrieved_data["chunks"]]
        
        # Get LLM generated answer
        result = rag_service_instance.answer_question(session_id, query)
        actual_output = result["answer"]

        # 2. Build DeepEval Test Case
        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieved_contexts
        )

        # 3. Measure Metrics
        self.faithfulness.measure(test_case)
        self.relevancy.measure(test_case)
        
        metrics_summary = {
            "strategy": strategy,
            "faithfulness_score": getattr(self.faithfulness, "score", 0.0),
            "relevancy_score": getattr(self.relevancy, "score", 0.0),
            "actual_output": actual_output
        }

        logger.info(f"DeepEval Results ({strategy}): {metrics_summary}")
        return metrics_summary

    def compare_all_strategies(
        self, 
        session_id: str, 
        query: str, 
        expected_output: str
    ) -> List[Dict[str, Any]]:
        """
        Compares all 4 strategies ('baseline', 'hybrid_rerank', 'fusion', 'crag') on the same test query.
        """
        strategies = ["baseline", "hybrid_rerank", "fusion", "crag"]
        comparison_results = []

        for strat in strategies:
            res = self.evaluate_query(session_id, query, expected_output, strategy=strat)
            comparison_results.append(res)

        return comparison_results

evaluator_instance = DeepEvalEvaluator()