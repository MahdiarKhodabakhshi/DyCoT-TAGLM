from __future__ import annotations
import json, re
from typing import Any, Dict, List
from tqdm import trange

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
IRI_RX    = re.compile(r"<\s*([^>\s]+)\s*>")
SELECT_RX = re.compile(r"\bSELECT\b", re.I)

def _replace_uri(match: re.Match) -> str:
    uri = match.group(1).strip()
    for ns, prefix in _PREFIX_ORDER:
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):].lstrip(':')}"
    return match.group(0)

def uri_collapse_after_select(sparql: str) -> str:
    m = SELECT_RX.search(sparql)
    if not m:
        return sparql
    body = sparql[m.start():].strip()
    return IRI_RX.sub(_replace_uri, body)

class QALDPreprocessor:
    def __init__(self, include_all_langs: bool = False):
        self.include_all_langs = include_all_langs

    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            questions = json.load(f)["questions"]
        for q in questions:
            if "new_query" not in q:
                raw = q.get("query", {}).get("sparql", "")
                q["new_query"] = uri_collapse_after_select(raw)
        return questions

    @staticmethod
    def _save(data: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _filter_english(self, qs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for q in qs:
            entry = {"id": q.get("id", ""), "question": "", "formated_query": q["new_query"], "answers": []}
            for phr in q.get("question", []):
                if phr.get("language") == "en":
                    entry["question"] = phr.get("string", "")
                if not self.include_all_langs:
                    break

            a0 = (q.get("answers") or [{}])[0]
            if "results" in a0:
                entry["answers"] = a0["results"]["bindings"]
            elif "boolean" in a0:
                entry["answers"] = [a0["boolean"]]
            out.append(entry)
        return out

    def run(self, in_path: str, out_path: str) -> List[Dict[str, Any]]:
        data = self._filter_english(self._load(in_path))
        self._save(data, out_path)
        return data

class LCQAPreprocessor:
    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def run(self, in_path: str, out_path: str) -> List[Dict[str, Any]]:
        proc: List[Dict[str, Any]] = []
        for rec in self._load(in_path):
            r = rec.copy()
            r["id"]        = r.pop("_id", r.get("id"))
            r["question"]  = r.pop("corrected_question", r.get("question"))
            raw_query      = r.pop("sparql_query", r.get("query"))
            r["query"]     = raw_query
            r["formated_query"] = uri_collapse_after_select(raw_query)
            norm: List[Dict[str, Dict[str, str]]] = []
            for a in r.get("answers") or []:
                if isinstance(a, dict):
                    norm.append(a)
                else:
                    norm.append({"callret-0": {"value": str(a)}})
            r["answers"] = norm
            proc.append(r)
        self._save(proc, out_path)
        return proc

class VQuandaPreprocessor:
    def _load(self, path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("VQuanda input must be a JSON array.")
        return data

    @staticmethod
    def _save(data: List[Dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _normalize(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            uid       = str(r.get("uid", ""))
            question  = r.get("question", "")
            raw_query = r.get("query", "")
            out.append({
                "id": uid,
                "question": question,
                "formated_query": uri_collapse_after_select(raw_query),
                "answers": [],
            })
        return out

    def run(self, in_path: str, out_path: str) -> List[Dict[str, Any]]:
        data = self._normalize(self._load(in_path))
        self._save(data, out_path)
        return data