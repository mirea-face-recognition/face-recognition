""" Custom Dataset

This file contains custom datasets to obtain data used in training.
"""
import json
from typing import Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from config import FACES_JSON, RECOGNITION_THRESHOLD


class FacesData:
    def __init__(self):
        self.path = Path(FACES_JSON)

    def load(self) -> Dict[str, List]:
        """Load all registered persons full names and their embeddings."""
        return json.loads(self.path.read_text(encoding='utf-8'))

    def add(self, name: str, embedding: List):
        """Add new users to JSON file.

        Args:
            name (str) - full name of person.
            embedding (List) - embedding of person's face."""
        registered_faces = self.load() if self.path.exists() else {}
        with open(self.path, 'w', encoding='utf-8') as f:
            registered_faces[name] = embedding
            json.dump(registered_faces, f, ensure_ascii=False)

    def get_most_similar(self, x) -> Optional[str]:
        """Use cosine similarity to find the most similar embedding.

        Args:
            x - embedding of incoming face.
        Returns:
            most_similar (str) - full name of most similar person."""
        registered_faces = self.load()
        if not registered_faces:
            return None
        similarities = {name: cosine_similarity([emb], [x]) for name, emb in registered_faces.items()}
        most_similar = max(similarities, key=similarities.get)
        return most_similar if similarities[most_similar] > RECOGNITION_THRESHOLD else None
