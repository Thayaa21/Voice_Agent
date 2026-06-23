"""
DEV-ONLY local stand-in for the Graph RAG backend (port 8000)
=============================================================
Lets you run and test Chetan's agent WITHOUT installing the heavy ML stack
(sentence-transformers / torch) or a running LLM.

It loads the REAL graph snapshot and serves the REAL backend functions:
  - graph_rag.api.smart_query.run_smart_query    (identity, conditions)
  - graph_rag.api.hospital_endpoints.*           (claim / cost / pre-proc / bill)

The only thing faked is the prose-synthesis LLM inside smart_query (a tiny stub).
The 4 hospital endpoints use no LLM, so their answers are identical to production.

Run from the repo root:
    python -m uvicorn chetan.dev_test_backend:app --port 8000
or from this folder:
    cd chetan && python -m uvicorn dev_test_backend:app --port 8000

Requires only: fastapi, uvicorn, networkx, rapidfuzz
"""
import sys, types, pickle
from pathlib import Path

# graph_rag_backend lives next to this folder, one level up.
HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "graph_rag_backend"
sys.path.insert(0, str(BACKEND))

# Stub the heavy core submodules so `graph_rag.core` imports without torch.
for name, attrs in {
    "graph_rag.core.embeddings": ["EmbeddingEngine"],
    "graph_rag.core.loader": ["DocumentLoader", "verify_line_offsets"],
    "graph_rag.core.graph_builder": ["KnowledgeGraphBuilder"],
    "graph_rag.core.resolver": ["EntityResolver"],
    "graph_rag.core.contradiction": ["ContradictionDetector"],
    "graph_rag.core.temporal": ["TemporalFilter"],
}.items():
    mod = types.ModuleType(name)
    for a in attrs:
        setattr(mod, a, type(a, (), {}))
    sys.modules[name] = mod

import networkx as nx  # noqa: F401
from fastapi import FastAPI
from pydantic import BaseModel
from graph_rag.api.smart_query import run_smart_query
from graph_rag.api import hospital_endpoints as hosp

SNAP = BACKEND / "graph_snapshot.pkl"
print(f"Loading snapshot from {SNAP} ...")
with open(SNAP, "rb") as f:
    GRAPH = pickle.load(f)["graph"]
print("Graph loaded:", GRAPH.number_of_nodes(), "nodes")

ENTITY_NODES = [(n, d) for n, d in GRAPH.nodes(data=True) if d.get("node_type") == "entity"]


class StubLLM:
    """smart_query calls llm.complete(prompt, temperature). A real model would
    summarise; here we surface the relevant facts so the agent has real content."""

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        if "Data for" not in prompt:
            return "Record found."
        block = prompt.split("Data for", 1)[1].split("Answer:", 1)[0]
        facts = [ln.strip() for ln in block.splitlines() if ":" in ln and "[" in ln]
        qline = ""
        parts = prompt.split("Question:", 1)
        if len(parts) > 1:
            qline = parts[1].split("\n", 1)[0].lower()
        if "condition" in qline or "diagnos" in qline:
            conds = [
                f.split("  [")[0].split(":", 1)[1].strip()
                for f in facts
                if f.lower().startswith("condition_description")
            ]
            if conds:
                return "Active conditions on file: " + "; ".join(dict.fromkeys(conds)) + "."
        keep = [f for f in facts if not f.lower().startswith(("condition_", "encounter_"))]
        return "; ".join(keep[:8]) if keep else (facts[0] if facts else "Record found.")


LLM = StubLLM()
app = FastAPI(title="DEV Graph RAG backend (snapshot replay)")


class Q(BaseModel):
    question: str
    max_hops: int = 3
    temporal_context: str = "current"


class HReq(BaseModel):
    patient_name: str


@app.get("/")
def root():
    return {"status": "ok", "service": "dev-graph-rag", "nodes": GRAPH.number_of_nodes()}


@app.post("/smart-query")
def smart_query(q: Q):
    return run_smart_query(q.question, GRAPH, LLM, ENTITY_NODES)


@app.post("/hospital/claim-status")
def claim_status(r: HReq):
    return hosp.get_claim_status(r.patient_name, GRAPH)


@app.post("/hospital/cost-estimate")
def cost_estimate(r: HReq):
    return hosp.get_cost_estimate(r.patient_name, GRAPH)


@app.post("/hospital/pre-procedure")
def pre_procedure(r: HReq):
    return hosp.get_pre_procedure_prep(r.patient_name, GRAPH)


@app.post("/hospital/bill-explanation")
def bill_explanation(r: HReq):
    return hosp.get_bill_explanation(r.patient_name, GRAPH)
