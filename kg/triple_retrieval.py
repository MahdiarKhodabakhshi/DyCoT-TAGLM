from __future__ import annotations
import json, os, re, time, urllib.error
from typing import Any, Dict, List, Tuple
from tqdm import trange
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQLJSON

PREFIX_MAP = {
    "http://dbpedia.org/ontology/": "dbo",
    "http://dbpedia.org/property/": "dbp",
    "http://dbpedia.org/resource/": "res",
    "http://dbpedia.org/class/yago/": "yago",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://www.w3.org/2001/XMLSchema#": "xsd",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "http://purl.org/dc/elements/1.1/": "dc",
    "http://purl.org/dc/terms/": "dcterms",
    "http://www.w3.org/2004/02/skos/core#": "skos",
    "http://www.w3.org/2003/01/geo/wgs84_pos#": "geo",
    "http://www.georss.org/georss/": "georss",
    "http://dbpedia.org/": "dbpedia",
    "http://purl.org/linguistics/gold/": "gold",
}
_PREFIX_ORDER = sorted(PREFIX_MAP.items(), key=lambda kv: -len(kv[0]))

def _shorten(uri: str) -> str:
    if uri.startswith("<") and uri.endswith(">"):
        uri = uri[1:-1]
    for ns, p in _PREFIX_ORDER:
        if uri.startswith(ns):
            return f"{p}:{uri[len(ns):].lstrip(':')}"
    return uri

class DBpediaRetriever:
    """
    For each entity (string like 'Paris'), fetch outgoing & incoming resource-only triples.
    Writes: sample['retrieved_triples'] = List[List[Tuple[str,str,str]]]
            (list per entity)
    Includes checkpointing to resume long jobs.
    """
    def __init__(
        self,
        endpoint: str = "https://dbpedia.org/sparql",
        timeout: int = 3000,
        max_retries: int = 5,
        retry_sleep: int = 10,
        checkpoint_file: str = "checkpoint.json",
        remove_checkpoint_on_complete: bool = True,
    ):
        self.endpoint  = endpoint
        self.timeout   = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.checkpoint_file = checkpoint_file
        self.remove_checkpoint_on_complete = remove_checkpoint_on_complete

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = self._load_checkpoint()
        done_ids = {rec["id"] for rec in existing}
        cache    = {rec["id"]: rec for rec in existing}

        for i in trange(len(data), desc="Retrieving DBpedia triples"):
            sample = data[i]
            sid    = sample.get("id")

            if sid in done_ids:
                data[i]["retrieved_triples"] = cache[sid]["retrieved_triples"]
                continue

            triples_by_entity: List[List[Tuple[str, str, str]]] = []
            for ent in sample.get("entities", []):
                cleaned = self._clean_uri(ent)
                if not cleaned:
                    triples_by_entity.append([("SKIPPED", "SKIPPED", _shorten(ent))])
                    continue
                triples = self._fetch_dbpedia_triples(cleaned)
                triples_by_entity.append(triples)

            sample["retrieved_triples"] = triples_by_entity
            cache[sid] = {"id": sid, "retrieved_triples": triples_by_entity}
            done_ids.add(sid)
            if i % 10 == 0:
                self._save_checkpoint(list(cache.values()))

        self._save_checkpoint(list(cache.values()))
        if self.remove_checkpoint_on_complete and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        return data

    # --- helpers ---
    def _fetch_dbpedia_triples(self, entity: str) -> List[Tuple[str, str, str]]:
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setTimeout(self.timeout)
        q = f"""
            SELECT ?subject ?predicate ?object WHERE {{
              {{ <http://dbpedia.org/resource/{entity}> ?predicate ?object .
                 FILTER(STRSTARTS(STR(?object), "http://dbpedia.org/resource/")) }}
              UNION
              {{ ?subject ?predicate <http://dbpedia.org/resource/{entity}> .
                 FILTER(STRSTARTS(STR(?subject), "http://dbpedia.org/resource/")) }}
            }}
        """
        sparql.setQuery(q)
        sparql.setReturnFormat(SPARQLJSON)

        for attempt in range(self.max_retries):
            try:
                res = sparql.query().convert()
                break
            except (urllib.error.HTTPError, urllib.error.URLError) as e:
                print(f"[Retry {attempt+1}/{self.max_retries}] {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_sleep)
                else:
                    return [("SKIPPED", "SKIPPED", _shorten(entity))]
            except Exception as e:
                print(f"[SPARQL error] {e}")
                return [("SKIPPED", "SKIPPED", _shorten(entity))]

        triples: List[Tuple[str, str, str]] = []
        for b in res["results"]["bindings"]:
            subj = b.get("subject",  {"value": f"http://dbpedia.org/resource/{entity}"} )["value"]
            pred = b.get("predicate",{"value": "UNKNOWN"})["value"]
            obj  = b.get("object",   {"value": f"http://dbpedia.org/resource/{entity}"} )["value"]
            triples.append((_shorten(subj), _shorten(pred), _shorten(obj)))
        return triples

    def _clean_uri(self, entity: str) -> str:
        entity = re.sub(r"[^\w\s-]", "", entity).replace(" ", "_")
        return entity.strip()

    def _load_checkpoint(self) -> List[Dict[str, Any]]:
        try:
            with open(self.checkpoint_file, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_checkpoint(self, results: List[Dict[str, Any]]) -> None:
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)