import os
import networkx as nx
from typing import List, Dict, Any
from backend.config import settings
from backend.utils.logger import logger

class GraphStoreManager:
    def __init__(self):
        pass

    def _get_graph_path(self, session_id: str) -> str:
        dir_path = os.path.join(settings.DATA_DIR, session_id)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, "knowledge_graph.graphml")

    def build_and_save_graph(self, session_id: str, triples_list: List[Dict[str, str]]) -> str:
        G = nx.DiGraph()

        for triple in triples_list:
            sub = triple.get("subject", "").strip()
            rel = triple.get("relation", "").strip()
            obj = triple.get("object", "").strip()

            if sub and rel and obj:
                G.add_node(sub, label="Entity")
                G.add_node(obj, label="Entity")
                G.add_edge(sub, obj, relation=rel)

        save_path = self._get_graph_path(session_id)
        nx.write_graphml(G, save_path)
        logger.info(f"Knowledge Graph saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges at: {save_path}")
        return save_path

    def load_graph(self, session_id: str) -> nx.DiGraph:
        save_path = self._get_graph_path(session_id)
        if not os.path.exists(save_path):
            return nx.DiGraph()
        return nx.read_graphml(save_path)

    def search_relationships(self, session_id: str, query: str) -> List[str]:
        """Finds subgraphs related to query words."""
        G = self.load_graph(session_id)
        if G.number_of_nodes() == 0:
            return []

        results = []
        query_words = set(query.lower().split())

        for u, v, data in G.edges(data=True):
            relation = data.get("relation", "CONNECTED_TO")
            edge_str = f"{u} --[{relation}]--> {v}"
            
            # Match query keywords against subject, object, or relation
            combined = f"{u} {relation} {v}".lower()
            if any(word in combined for word in query_words if len(word) > 2):
                results.append(edge_str)

        return results[:10]  # Return top 10 relevant relationship strings