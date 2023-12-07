""" Custom Dataset

This file contains custom datasets to obtain data used in training.
"""
import json
from typing import Dict, List

from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from src.config import FACES_JSON


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

    def get_most_similar(self, x) -> str:
        """Use cosine similarity to find the most similar face.

        Args:
            x - embedding of incoming face.
        Returns:
            most_similar (str) - full name of most similar person."""
        # TODO add threshold for unrecognized faces
        registered_faces = self.load()
        most_similar, = max(registered_faces.items(), key=lambda e: cosine_similarity([e[1]], [x]))
        return most_similar
