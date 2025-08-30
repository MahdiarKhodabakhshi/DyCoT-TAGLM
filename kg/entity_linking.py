from __future__ import annotations
from typing import List, Dict, Any
from tqdm import tqdm

class EntityExtractorBase:
    def extract_entities(self, text: str) -> List[str]:
        raise NotImplementedError

    def run(self, data: List[Dict[str, Any]], question_key: str = "question",
            out_key: str = "entities") -> List[Dict[str, Any]]:
        for i in tqdm(range(len(data)), desc=f"Entity Extraction · {self.__class__.__name__}"):
            q = data[i].get(question_key, "")
            ents = self.extract_entities(q)
            data[i][out_key] = ents
        return data

# ---- ReFinED (primary) ----
class RefinedEntityExtractor(EntityExtractorBase):
    def __init__(self, refined_model):
        self.refined_model = refined_model

    def extract_entities(self, text: str) -> List[str]:
        spans = self.refined_model.process_text(text)
        entities = [
            span.predicted_entity.wikipedia_entity_title.replace(' ', '_')
            for span in spans if getattr(getattr(span, "predicted_entity", None),
                                         "wikipedia_entity_title", None)
        ]
        return entities

# ---- Falcon (ablation) ----
class FalconEntityExtractor(EntityExtractorBase):
    def __init__(self, api_url: str = "https://labs.tib.eu/falcon/api?mode=long",
                 delay: float = 0.5, timeout: int = 30):
        self.api_url = api_url
        self.delay = delay
        self.timeout = timeout
        self.headers = {'Content-Type': 'application/json'}

    def extract_entities(self, text: str) -> List[str]:
        import requests, time
        payload = {"text": text}
        try:
            res = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(f"[Falcon] error for '{text[:50]}...': {e}")
            return []

        raw = data.get('entities', [])
        uris: List[str] = []
        if raw:
            if isinstance(raw[0], dict):
                uris = [ent.get("URI") for ent in raw if ent.get("URI")]
            elif all(isinstance(ent, str) for ent in raw):
                uris = raw

        entities = []
        for uri in uris:
            if uri and "/" in uri:
                entities.append(uri.split('/')[-1])
            else:
                entities.append(uri)
        time.sleep(self.delay)
        return entities