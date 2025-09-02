from __future__ import annotations
import argparse
from pathlib import Path

from dycot_taglm.utils.io import save_json, load_json
from dycot_taglm.dataio.preprocessors import QALDPreprocessor, LCQAPreprocessor, VQuandaPreprocessor
from dycot_taglm.kg.sparql_parser import SparqlParser, SparqlParserLCQuad
from dycot_taglm.kg.entity_linking import RefinedEntityExtractor, FalconEntityExtractor
from dycot_taglm.kg.triple_retrieval import DBpediaRetriever

# NOTE: install refined and model assets as you already do in your environment:
# from refined.inference.processor import Refined

def build_qald_flow(in_raw: str, workdir: str, entity_extractor: str = "refined") -> str:
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    # 1) preprocess
    pre_out = work / "qald9_en.json"
    data = QALDPreprocessor(include_all_langs=False).run(in_raw, str(pre_out))

    # 2) parse SPARQL
    data = SparqlParser().run(data)
    save_json(data, work / "qald9_triples.json")

    # 3) entity extraction
    if entity_extractor == "falcon":
        extractor = FalconEntityExtractor()
    else:
        from refined.inference.processor import Refined
        refined_model = Refined.from_pretrained(model_name="wikipedia_model", entity_set="wikipedia")
        extractor = RefinedEntityExtractor(refined_model)

    data = extractor.run(data)
    save_json(data, work / f"qald9_entities_{entity_extractor}.json")

    # 4) DBpedia triples
    retr = DBpediaRetriever(checkpoint_file=str(work / "checkpoint_qald.json"))
    data = retr.run(data)
    out_path = work / f"qald9_retrieved_triples_{entity_extractor}.json"
    save_json(data, out_path)
    return str(out_path)

def build_lcquad_flow(in_raw: str, workdir: str, entity_extractor: str = "refined") -> str:
    work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
    data = LCQAPreprocessor().run(in_raw, str(work / "lcquad_en.json"))
    data = SparqlParserLCQuad().run(data)
    save_json(data, work / "lcquad_triples.json")

    if entity_extractor == "falcon":
        extractor = FalconEntityExtractor()
    else:
        from refined.inference.processor import Refined
        refined_model = Refined.from_pretrained(model_name="wikipedia_model", entity_set="wikipedia")
        extractor = RefinedEntityExtractor(refined_model)
    data = extractor.run(data)
    save_json(data, work / f"lcquad_entities_{entity_extractor}.json")

    retr = DBpediaRetriever(checkpoint_file=str(work / "checkpoint_lcquad.json"))
    data = retr.run(data)
    out_path = work / f"lcquad_retrieved_triples_{entity_extractor}.json"
    save_json(data, out_path)
    return str(out_path)

def build_vquanda_flow(in_raw: str, workdir: str, entity_extractor: str = "refined") -> str:
    work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
    data = VQuandaPreprocessor().run(in_raw, str(work / "vquanda_norm.json"))
    data = SparqlParser().run(data)
    save_json(data, work / "vquanda_triples.json")

    if entity_extractor == "falcon":
        extractor = FalconEntityExtractor()
    else:
        from refined.inference.processor import Refined
        refined_model = Refined.from_pretrained(model_name="wikipedia_model", entity_set="wikipedia")
        extractor = RefinedEntityExtractor(refined_model)
    data = extractor.run(data)
    save_json(data, work / f"vquanda_entities_{entity_extractor}.json")

    retr = DBpediaRetriever(checkpoint_file=str(work / "checkpoint_vquanda.json"))
    data = retr.run(data)
    out_path = work / f"vquanda_retrieved_triples_{entity_extractor}.json"
    save_json(data, out_path)
    return str(out_path)

def main():
    ap = argparse.ArgumentParser(description="Build datasets with retrieved & ranked triples (ranking run is separate).")
    ap.add_argument("--task", choices=["qald9", "lcquad", "vquanda"], required=True)
    ap.add_argument("--in_raw", required=True, help="Path to the raw dataset file")
    ap.add_argument("--workdir", required=True, help="Working directory for intermediates & outputs")
    ap.add_argument("--entities", choices=["refined", "falcon"], default="refined")
    args = ap.parse_args()

    if args.task == "qald9":
        out = build_qald_flow(args.in_raw, args.workdir, args.entities)
    elif args.task == "lcquad":
        out = build_lcquad_flow(args.in_raw, args.workdir, args.entities)
    else:
        out = build_vquanda_flow(args.in_raw, args.workdir, args.entities)

    print(f"✅ retrieved triples written: {out}\n"
          f"➡️  next: run ColBERT ranking on {out}")

if __name__ == "__main__":
    main()