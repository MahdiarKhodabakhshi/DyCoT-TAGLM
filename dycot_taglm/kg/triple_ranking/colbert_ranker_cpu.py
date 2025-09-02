from __future__ import annotations
import os, json, random
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("TORCH_NUM_THREADS", "8")
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "2")

import torch
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "8")))

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher

def _clean_tok(x: Any) -> str:
    return str(x).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()

def as_text(equiv: List[Any]) -> str:
    s = " ".join(_clean_tok(x) for x in equiv)
    return " ".join(s.split())

def _equiv_key(equiv: List[Any]) -> Tuple[str, str, str]:
    t = tuple(_clean_tok(x) for x in equiv)
    if len(t) != 3:
        t = (t + ("", "", ""))[:3]
    return t

def build_global_collection(
    data: List[Dict[str, Any]],
    collection_tsv: str
):
    Path(collection_tsv).parent.mkdir(parents=True, exist_ok=True)

    equiv2docid: Dict[Tuple[str, str, str], int] = {}
    docid2qids: Dict[int, set] = defaultdict(set)
    docid2payload_by_qid: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    did, written = 0, 0
    with open(collection_tsv, "w", encoding="utf-8") as fout:
        for entry in data:
            q_id = int(entry.get("id"))
            for item in entry.get("retrieved_triples", []):
                if isinstance(item, dict):
                    if "equivalent" not in item or "triple" not in item:
                        continue
                    equiv = item["equivalent"]
                    triple = item["triple"]
                else:
                    continue

                if not isinstance(equiv, (list, tuple)) or len(equiv) != 3:
                    continue

                key = _equiv_key(list(equiv))
                if key not in equiv2docid:
                    equiv2docid[key] = did
                    text = as_text(list(equiv))
                    fout.write(f"{did}\t{text}\n")
                    did += 1
                    written += 1

                doc_id = equiv2docid[key]
                docid2qids[doc_id].add(q_id)
                docid2payload_by_qid[doc_id][q_id] = {"triple": triple, "equivalent": list(equiv)}

    return equiv2docid, docid2qids, docid2payload_by_qid, did, written

def index_collection(
    collection_tsv: str,
    experiment: str,
    index_name: str = "triples.nbits=2",
    root_dir: str = "experiments",
    checkpoint: str = "colbertv2.0",
    doc_maxlen: int | None = None,
    overwrite_index: bool = True
):
    with Run().context(RunConfig(nranks=1, experiment=experiment)):
        cfg_kwargs = dict(nbits=2, root=root_dir, amp=False)
        if doc_maxlen is not None:
            cfg_kwargs["doc_maxlen"] = doc_maxlen
        config = ColBERTConfig(**cfg_kwargs)

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(
            name=index_name,
            collection=collection_tsv,
            overwrite=("force_silent_overwrite" if overwrite_index else False),
        )
        index_path = os.path.join(config.root, experiment, "indexes", index_name)
    return index_path, config

def run_colbert_ranking(
    input_json: str,
    output_json: str,
    collection_tsv: str,
    experiment: str,
    index_name: str = "triples.nbits=2",
    root_dir: str = "experiments",
    checkpoint: str = "colbertv2.0",
    top_k: int = 100,
    seed: int = 42
) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    data: List[Dict[str, Any]] = json.loads(Path(input_json).read_text(encoding="utf-8"))

    equiv2docid, docid2qids, docid2payload_by_qid, total_docs, written = build_global_collection(
        data, collection_tsv
    )
    print(f"[collection] wrote {written} unique docs to {collection_tsv}")

    index_path, base_config = index_collection(
        collection_tsv=collection_tsv,
        experiment=experiment,
        index_name=index_name,
        root_dir=root_dir,
        checkpoint=checkpoint,
        overwrite_index=True
    )
    print(f"[index] ready at {index_path}")

    searcher = Searcher(index=index_path, collection=collection_tsv, config=base_config)

    for entry in data:
        q_id = int(entry["id"])
        query = entry.get("question", "")
        doc_ids, ranks, scores = searcher.search(query, k=top_k)
        rel = [(did, rnk, sc) for did, rnk, sc in zip(doc_ids, ranks, scores) if q_id in docid2qids.get(did, set())]
        rel.sort(key=lambda x: x[2], reverse=True)

        ranked = []
        for final_rank, (did, orig_rank, score) in enumerate(rel, 1):
            payload = docid2payload_by_qid[did][q_id]
            ranked.append({
                "triple": payload["triple"],
                "equivalent": payload["equivalent"],
                "score": float(score),
                "original_colbert_rank": int(orig_rank),
                "final_rank": int(final_rank),
            })
        entry["retrieved_triples_ranked"] = ranked

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote ranked triples to {output_json}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--collection_tsv", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--index_name", default="triples.nbits=2")
    p.add_argument("--root_dir", default="experiments")
    p.add_argument("--checkpoint", default="colbertv2.0")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run_colbert_ranking(**vars(args))
