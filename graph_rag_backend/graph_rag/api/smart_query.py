"""
Smart Query Engine
==================
Intelligent natural-language query over the knowledge graph.

Features:
- Fuzzy name matching (handles typos: "mai lee" → "Mei Lee")
- Disambiguation when multiple people match ("james" → 3 James found)
- "Did you mean?" for near-misses
- LLM synthesis with document provenance
- Open-ended aggregate queries ("how many licenses?")
"""

import concurrent.futures
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


def _canonical(name: str) -> str:
    """Strip middle names/initials: 'James Robert Lee' → 'james lee'"""
    parts = name.strip().lower().split()
    if len(parts) >= 3:
        return f"{parts[0]} {parts[-1]}"
    return " ".join(parts)


def _fuzzy_score(query_phrase: str, canon_name: str) -> float:
    """
    Score how well query_phrase matches canon_name.
    Prioritizes full-phrase similarity over partial-token overlap.
    """
    a = query_phrase.lower().strip()
    b = canon_name.lower().strip()

    if a == b:
        return 1.0

    try:
        from rapidfuzz import fuzz
        # WRatio: best of several strategies (partial, token sort, etc.)
        wr = fuzz.WRatio(a, b) / 100.0
        # Partial ratio: a contained in b or vice versa
        pr = fuzz.partial_ratio(a, b) / 100.0
        # Prefer WRatio heavily — it's the most balanced
        return 0.7 * wr + 0.3 * pr
    except ImportError:
        if a in b or b in a:
            return 0.8
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        shared = tokens_a & tokens_b
        return len(shared) / max(len(tokens_a), len(tokens_b), 1)


def run_smart_query(question_raw: str, graph, llm, entity_nodes) -> dict:
    """
    Main entry point. Returns a response dict with 'type' field:
      - 'empty'           — no data in graph
      - 'summary'         — open-ended aggregate question
      - 'disambiguation'  — multiple people match
      - 'suggestion'      — fuzzy near-miss, asking "did you mean?"
      - 'not_found'       — no match at all
      - 'answer'          — single person, full LLM answer + facts
    """
    question_lower = question_raw.lower().strip()

    if not entity_nodes:
        return {"type": "empty", "answer": "No entities in graph.", "options": []}

    # ── Build canonical person map ─────────────────────────────────────────
    canonical_map: dict[str, list] = {}   # canon_name → [(nid, data)]
    for nid, data in entity_nodes:
        raw_name = data.get("name", "")
        if not raw_name:
            continue
        canon = _canonical(raw_name)
        canonical_map.setdefault(canon, []).append((nid, data))

    # ── Open-ended detection ───────────────────────────────────────────────
    OPEN_ENDED_KW = [
        "how many", "list all", "show all", "count", "every", "total",
        "all people", "all entities", "all records", "what documents",
        "summarize everyone", "how many licenses", "how many passports",
    ]
    if any(kw in question_lower for kw in OPEN_ENDED_KW):
        return _open_ended(question_raw, canonical_map, entity_nodes, llm)

    # ── Extract query phrases ──────────────────────────────────────────────
    STOP = {
        "what", "is", "are", "the", "of", "get", "tell", "me", "about",
        "give", "find", "show", "list", "their", "his", "her", "for",
        "who", "does", "do", "has", "have", "a", "an", "from", "by", "with", "at",
        "how", "when", "where", "why", "did", "was", "were", "will", "can",
    }
    # strip possessives and stopwords
    clean_q = re.sub(r"[''`]s?\b", " ", question_lower)
    q_words = [w.strip(".,!?") for w in clean_q.split() if w.strip(".,!?") not in STOP and len(w) >= 3]

    candidates: list[str] = list(q_words)  # single words
    for i in range(len(q_words) - 1):      # adjacent pairs
        candidates.append(f"{q_words[i]} {q_words[i+1]}")

    # ── Score each canonical name ──────────────────────────────────────────
    # Score multi-word candidates first; single words only if no multi-word
    # candidate was relevant. This prevents "mei" alone from matching "mei johnson"
    # when the full phrase "mei lee" is present.
    multi_scores: dict[str, float] = {}
    single_scores: dict[str, float] = {}

    for candidate in candidates:
        is_multi = " " in candidate
        target   = multi_scores if is_multi else single_scores
        for canon in canonical_map:
            s = _fuzzy_score(candidate, canon)
            if is_multi:
                s = min(1.0, s * 1.10)
            if s > target.get(canon, 0.0):
                target[canon] = s

    # Use multi-word scores where available; fall back to single-word scores
    name_scores: dict[str, float] = {}
    if multi_scores:
        name_scores = multi_scores
    else:
        name_scores = single_scores

    STRONG = 0.75
    WEAK   = 0.50

    strong = {n: s for n, s in name_scores.items() if s >= STRONG}
    weak   = {n: s for n, s in name_scores.items() if WEAK <= s < STRONG}

    # If we have a clear winner, prune weaker matches
    if len(strong) > 1:
        best_score = max(strong.values())
        strong = {n: s for n, s in strong.items() if best_score - s <= 0.10}

    # ── No match ──────────────────────────────────────────────────────────
    if not strong and not weak:
        top3 = sorted(name_scores.items(), key=lambda x: -x[1])[:3]
        suggestions = [n.title() for n, s in top3 if s > 0.25]
        msg = "No matching people found."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        return {"type": "not_found", "answer": msg, "suggestions": suggestions, "options": []}

    # ── Only weak matches → "Did you mean?" ───────────────────────────────
    if not strong and weak:
        best = sorted(weak.items(), key=lambda x: -x[1])[0][0]
        return {
            "type":        "suggestion",
            "answer":      f"I found a possible match: **{best.title()}**. Is that who you're looking for?",
            "suggestions": [n.title() for n in sorted(weak.keys(), key=lambda x: -weak[x])],
            "options":     [],
        }

    # ── Multiple strong matches → disambiguation ──────────────────────────
    if len(strong) > 1:
        options = []
        for canon in sorted(strong.keys()):
            entities = canonical_map[canon]
            doc_types = sorted({d.get("doc_type", "") for _, d in entities})
            src_files = sorted({d.get("source_filename", "") for _, d in entities})
            attrs = entities[0][1].get("attributes", {}) or {} if entities else {}
            options.append({
                "name":       canon.title(),
                "doc_types":  doc_types,
                "files":      src_files,
                "dob_hint":   attrs.get("dob", ""),
                "entity_ids": [nid for nid, _ in entities],
            })
        return {
            "type":    "disambiguation",
            "answer":  f"I found {len(strong)} people matching your query. Which one do you mean?",
            "options": options,
        }

    # ── Single match → full answer ────────────────────────────────────────
    canon = list(strong.keys())[0]
    match_score = strong[canon]
    return _single_person_answer(question_raw, canon, match_score, canonical_map[canon], graph, llm)


def _open_ended(question_raw: str, canonical_map: dict, entity_nodes: list, llm) -> dict:
    doc_type_counts: dict = {}
    for _, data in entity_nodes:
        dt = data.get("doc_type", "GENERIC")
        doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1

    context = (
        f"Database has {len(canonical_map)} unique people, "
        f"{len(entity_nodes)} total records.\n"
        f"Document types: {doc_type_counts}\n"
        f"People: {', '.join(sorted(canonical_map.keys()))}\n"
    )
    prompt = (
        f"Answer concisely. Use the database summary below.\n\n"
        f"Question: {question_raw}\n\nDatabase:\n{context}\n\nAnswer:"
    )
    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            answer = ex.submit(llm.complete, prompt, 0.0).result(timeout=20)
    except Exception:
        answer = (
            f"The database contains {len(canonical_map)} people: "
            f"{', '.join(sorted(canonical_map.keys())[:15])}..."
        )
    return {
        "type":     "summary",
        "answer":   answer,
        "count":    len(canonical_map),
        "entities": sorted(canonical_map.keys()),
        "options":  [],
    }


def _single_person_answer(question_raw: str, canon: str, match_score: float,
                           entities: list, graph, llm) -> dict:
    all_facts: list[dict] = []
    context_lines: list[str] = []

    for nid, data in entities:
        attrs    = data.get("attributes", {}) or {}
        src_file = data.get("source_filename", "")
        doc_type = data.get("doc_type", "")
        line_num, line_text = 0, ""

        for neighbor in graph.neighbors(nid):
            edge = graph.edges.get((nid, neighbor)) or graph.edges.get((neighbor, nid), {})
            if edge.get("edge_type") == "mentions":
                line_num  = edge.get("line_number", 0)
                line_text = edge.get("line_text", "")
                break

        for key, val in attrs.items():
            if key.startswith("_") or not val:
                continue
            src_tag = src_file + (f" line {line_num}" if line_num else "")
            all_facts.append({
                "fact":            f"{key}: {val}",
                "source_filename": src_file,
                "doc_type":        doc_type,
                "line_number":     line_num,
                "line_text":       line_text,
                "attribute_key":   key,
                "value":           str(val),
            })
            context_lines.append(f"{key}: {val}  [{src_tag}]")

    if not all_facts:
        return {
            "type":   "not_found",
            "answer": f"Found {canon.title()} in the graph but no attributes extracted.",
            "options": [],
        }

    context_str = "\n".join(context_lines[:40])
    clarification = ""
    if match_score < 0.90:
        clarification = (
            f"(Note: matched '{question_raw}' to '{canon.title()}' in the database — "
            f"if this is wrong, please be more specific.)\n\n"
        )

    prompt = (
        f"You are answering a question about people in a document database.\n"
        f"Answer specifically and concisely, mentioning which document each fact comes from.\n\n"
        f"{clarification}"
        f"Question: {question_raw}\n\n"
        f"Data for {canon.title()}:\n{context_str}\n\n"
        f"Answer:"
    )
    try:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            answer = ex.submit(llm.complete, prompt, 0.0).result(timeout=25)
    except Exception:
        answer = _fallback_answer(question_raw, canon, all_facts)

    # Find conflicts
    entity_ids = {nid for nid, _ in entities}
    conflicts = []
    for u, v, edata in graph.edges(data=True):
        if edata.get("edge_type") == "conflict" and (u in entity_ids or v in entity_ids):
            nu = graph.nodes.get(u, {})
            nv = graph.nodes.get(v, {})
            conflicts.append({
                "conflict_type": edata.get("conflict_type", ""),
                "attribute_key": edata.get("attribute_key", ""),
                "value_a":       edata.get("value_a", ""),
                "value_b":       edata.get("value_b", ""),
                "severity":      edata.get("severity", "minor"),
                "source_doc_a":  nu.get("source_filename", ""),
                "source_doc_b":  nv.get("source_filename", ""),
            })

    return {
        "type":          "answer",
        "person":        canon.title(),
        "answer":        answer,
        "facts":         all_facts,
        "conflicts":     conflicts,
        "has_conflicts": len(conflicts) > 0,
        "options":       [],
    }


def _fallback_answer(question: str, canon: str, all_facts: list) -> str:
    q = question.lower()
    kw_map = {
        frozenset(["dob", "birth", "born", "birthday"]): "dob",
        frozenset(["license", "licence", "driving"]):    "license_number",
        frozenset(["passport"]):                         "passport_number",
        frozenset(["address", "live", "home"]):          "address",
        frozenset(["insurance", "policy"]):              "policy_number",
        frozenset(["diagnosis", "medical", "condition"]): "diagnosis",
        frozenset(["medication", "drug", "prescription"]): "medications",
    }
    relevant = []
    for kws, attr in kw_map.items():
        if any(kw in q for kw in kws):
            relevant += [f for f in all_facts if f["attribute_key"] == attr]
    if not relevant:
        relevant = all_facts[:5]

    parts = []
    for f in relevant[:5]:
        src = f["source_filename"]
        if f["line_number"]:
            src += f" line {f['line_number']}"
        parts.append(f"{f['attribute_key']}: {f['value']}  [{src}]")
    return f"{canon.title()} — " + " | ".join(parts)
