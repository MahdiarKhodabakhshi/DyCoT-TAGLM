from __future__ import annotations
import re
from typing import Any, Dict, List
from tqdm import trange

STRING_RE = r'"(?:[^"\\]|\\.)*"(?:@[A-Za-z\-]+|\^\^[^\s;,.{}()]+)?'
IRI_RE    = r'<[^>]*>'
PNAME_RE  = r'[^\s;,.{}()]+'
SEP_RE    = r'[;,.]'
TOKEN_RE  = rf'(?:{STRING_RE}|{IRI_RE}|{PNAME_RE}|{SEP_RE})'

PREFIX_PATTERN = re.compile(r'PREFIX\s+([a-z0-9]+):\s*<([^>]+)>', re.I)
WHERE_PATTERN  = re.compile(r'WHERE\s*\{(.+?)\}', re.I | re.S)

def _extract_triples(block_text: str) -> List[List[str]]:
    strip = [
        r'FILTER\s*\([^)]*\)', r'BIND\s*\([^)]*\)', r'GROUP\s+BY[^.}]*',
        r'HAVING[^.}]*', r'ORDER\s+BY[^.}]*', r'LIMIT\s+\d+', r'OFFSET\s+\d+',
    ]
    for pat in strip:
        block_text = re.sub(pat, '', block_text, flags=re.I | re.S)

    tokens, triples, subj, pred = re.findall(TOKEN_RE, block_text), [], None, None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == '.':
            subj = pred = None
        elif t == ';':
            pred = None
        elif t == ',':
            pass
        elif subj is None:
            subj = t
        elif pred is None:
            pred = t
        else:
            triples.append([subj, pred, t])
        i += 1
    return triples

def _top_level_blocks(where_text: str) -> List[str]:
    blocks, buf, depth = [], [], 0
    for ch in where_text:
        if ch == "{":
            if depth == 0 and buf:
                blocks.append("".join(buf).strip()); buf = []
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                buf.append(ch)
            else:
                blocks.append("".join(buf).strip()); buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        blocks.append(tail)
    return blocks

class SparqlParser:
    def __init__(self) -> None:
        self.global_prefixes: Dict[str, str] = {}

    def parse_sparql(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx in trange(len(samples), desc="Parsing SPARQL"):
            s          = samples[idx]
            query_text = s.get("formated_query") or s.get("query") or ""

            ans_map: Dict[str, List[str]] = {}
            for a in s.get("answers", []):
                if isinstance(a, bool):
                    continue
                v = next(iter(a))
                ans_map.setdefault(v, []).append(a[v]["value"])
            s["answers_value"] = ans_map

            local = dict(PREFIX_PATTERN.findall(query_text))
            for k, v in local.items():
                self.global_prefixes.setdefault(k, v)

            m = WHERE_PATTERN.search(query_text)
            if not m:
                s["triples"] = []
                continue
            where_txt = m.group(1)

            flat = []
            for block in _top_level_blocks(where_txt):
                flat.extend(_extract_triples(block))

            expanded = []
            for tri in flat:
                exp = []
                for tok in tri:
                    p, *tail = tok.split(":", 1)
                    exp.append(f"{local[p]}:{tail[0]}" if tail and p in local else tok)
                expanded.append(exp)
            s["triples"] = expanded
        return samples

    def run(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.parse_sparql(data)

class SparqlParserLCQuad(SparqlParser):
    def parse_sparql(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for s in samples:
            if isinstance(s.get("answers"), dict):
                s["answers"] = [s["answers"]]
        return super().parse_sparql(samples)