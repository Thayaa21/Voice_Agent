"""
Hospital Voice Agent Endpoints
================================
Four dedicated endpoints for the 4 high-value use cases:

  POST /hospital/claim-status      → prior auth denials, pending P2P
  POST /hospital/cost-estimate     → out-of-pocket calculation
  POST /hospital/pre-procedure     → prep instructions before a procedure
  POST /hospital/bill-explanation  → post-service bill dispute resolution

Each endpoint:
  1. Fuzzy-matches the patient name in the graph
  2. Pulls the relevant structured data from their entities
  3. Returns a spoken-ready natural language answer

All responses follow the same contract:
  {
    "type":         "answer" | "not_found" | "disambiguation",
    "person":       "Display Name",
    "answer":       "Text to speak to the caller",
    "has_conflicts": false,
    "options":      []   (populated on disambiguation)
  }
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _canonical(name: str) -> str:
    parts = name.strip().lower().split()
    if len(parts) >= 3:
        return f"{parts[0]} {parts[-1]}"
    return " ".join(parts)


def _fuzzy_match_person(patient_name: str, graph) -> tuple[str | None, list]:
    """
    Find the best matching patient in the graph.
    Returns (canonical_name, list_of_entity_nodes) or (None, []).
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        fuzz = None

    query = patient_name.strip().lower()

    # Build canonical → entities map (only primary PERSON entities with insurance_payer)
    person_map: dict[str, list] = {}
    for nid, data in graph.nodes(data=True):
        if data.get("node_type") != "entity":
            continue
        attrs = data.get("attributes", {}) or {}
        # Primary person entity has insurance_payer field
        if "insurance_payer" not in attrs:
            continue
        raw = data.get("name", "")
        if not raw:
            continue
        canon = _canonical(raw)
        person_map.setdefault(canon, []).append((nid, data))

    if not person_map:
        return None, []

    # Score each canonical name
    best_name, best_score = None, 0.0
    for canon in person_map:
        if fuzz:
            score = fuzz.WRatio(query, canon) / 100.0
        else:
            score = 1.0 if query == canon else (0.8 if query in canon or canon in query else 0.0)
        if score > best_score:
            best_score, best_name = score, canon

    if best_score < 0.60:
        return None, []

    return best_name, person_map[best_name]

def _get_all_entities_for_person(patient_name: str, graph) -> tuple[str | None, list]:
    """
    Get ALL entities (person + conditions + encounters) for a matched patient.
    """
    canon, primary_entities = _fuzzy_match_person(patient_name, graph)
    if not canon:
        return None, []

    # Get the display name from the first primary entity
    display_name = primary_entities[0][1].get("name", canon.title()) if primary_entities else canon.title()

    # Now get ALL entities for this person (conditions + encounters)
    all_entities = []
    for nid, data in graph.nodes(data=True):
        if data.get("node_type") != "entity":
            continue
        if _canonical(data.get("name", "")) == canon:
            all_entities.append((nid, data))

    return display_name, all_entities


def _not_found(name: str) -> dict:
    return {
        "type":          "not_found",
        "person":        name,
        "answer":        f"I couldn't find a patient named {name} in our records. Could you please repeat your full name and date of birth?",
        "has_conflicts": False,
        "options":       [],
    }


def _answer(person: str, text: str) -> dict:
    return {
        "type":          "answer",
        "person":        person,
        "answer":        text,
        "has_conflicts": False,
        "options":       [],
    }


# ---------------------------------------------------------------------------
# 1. CLAIM STATUS — prior auth denials, pending P2P
# ---------------------------------------------------------------------------

def get_claim_status(patient_name: str, graph) -> dict:
    """
    Use case: "Why was my surgery denied?" / "What's the status of my claim?"

    Finds the patient's most recent or most problematic claim and explains
    the status in plain English.
    """
    display_name, all_entities = _get_all_entities_for_person(patient_name, graph)
    if not display_name:
        return _not_found(patient_name)

    # Collect encounter entities
    encounters = []
    primary_attrs = {}
    for _, data in all_entities:
        attrs = data.get("attributes", {}) or {}
        if attrs.get("record_type") == "encounter":
            encounters.append(attrs)
        elif "insurance_payer" in attrs:
            primary_attrs = attrs

    if not encounters:
        return _answer(display_name, f"I don't have any encounter records on file for {display_name}.")

    # Find the most important claim — prioritize DENIED, then PENDING_P2P, then most recent
    priority = {"DENIED_NO_PA": 0, "PENDING_P2P": 1, "PARTIAL": 2, "PAID": 3}
    encounters_sorted = sorted(
        encounters,
        key=lambda e: (priority.get(e.get("claim_status", "PAID"), 3), e.get("encounter_date", "") or "")
    )
    top = encounters_sorted[0]

    status      = top.get("claim_status", "PAID")
    date        = top.get("encounter_date", "unknown date")
    description = top.get("description", "your procedure")
    reason      = top.get("reason", "")
    billed      = float(top.get("billed", 0) or 0)
    covered     = float(top.get("covered", 0) or 0)
    owes        = float(top.get("patient_owes", 0) or 0)
    payer       = top.get("payer", primary_attrs.get("insurance_payer", "your insurance"))

    # Claim summary
    denied_count  = int(primary_attrs.get("denied_claims", 0))
    pending_count = int(primary_attrs.get("pending_claims", 0))

    if status == "DENIED_NO_PA":
        msg = (
            f"{display_name}, your {description} claim from {date} was denied — prior authorization not obtained. "
            f"Billed: ${billed:,.2f}, covered: $0. "
            f"To appeal: your care team submits a prior auth request to {primary_attrs.get('insurance_payer','your insurer')}. "
            f"If denied again, your doctor can request a Peer-to-Peer review."
        )
        if denied_count > 1:
            msg += f" You have {denied_count} total denied claims."

    elif status == "PENDING_P2P":
        msg = (
            f"{display_name}, your {description} claim from {date} is pending a Peer-to-Peer review with {payer}. "
            f"Your doctor needs to speak with the insurer's medical director. "
            f"Billed: ${billed:,.2f}, potentially covered: ${covered:,.2f}. Typically resolves in 3-5 business days."
        )

    elif status == "PARTIAL":
        msg = (
            f"{display_name}, your {description} claim from {date} was partially covered. "
            f"{payer} paid ${covered:,.2f} of ${billed:,.2f}. Your balance: ${owes:,.2f}."
        )

    else:  # PAID
        msg = (
            f"{display_name}, your most recent claim for {description} on {date} was fully processed. "
            f"{payer} covered ${covered:,.2f} of the ${billed:,.2f} billed. "
        )
        if denied_count > 0:
            msg += f" However, you do have {denied_count} other denied claim{'s' if denied_count > 1 else ''} on your account that may need attention."
        else:
            msg += "All your recent claims appear to be in order."

    return _answer(display_name, msg)


# ---------------------------------------------------------------------------
# 2. COST ESTIMATE — out-of-pocket calculation
# ---------------------------------------------------------------------------

def get_cost_estimate(patient_name: str, graph) -> dict:
    """
    Use case: "How much will I owe for my upcoming procedure?"

    Calculates the patient's financial picture based on their claim history.
    """
    display_name, all_entities = _get_all_entities_for_person(patient_name, graph)
    if not display_name:
        return _not_found(patient_name)

    primary_attrs = {}
    encounters = []
    for _, data in all_entities:
        attrs = data.get("attributes", {}) or {}
        if "insurance_payer" in attrs:
            primary_attrs = attrs
        elif attrs.get("record_type") == "encounter":
            encounters.append(attrs)

    payer        = primary_attrs.get("insurance_payer", "your insurance")
    total_billed = float(primary_attrs.get("total_billed", 0))
    total_covered = float(primary_attrs.get("total_covered", 0))
    total_owed   = float(primary_attrs.get("total_owed", 0))
    denied_count = int(primary_attrs.get("denied_claims", 0))

    # Calculate average coverage ratio from paid encounters
    paid_encounters = [e for e in encounters if e.get("claim_status") == "PAID" and float(e.get("billed", 0)) > 0]
    if paid_encounters:
        avg_coverage = sum(float(e.get("covered", 0)) / float(e.get("billed", 1)) for e in paid_encounters) / len(paid_encounters)
        coverage_pct = round(avg_coverage * 100)
    else:
        coverage_pct = 0

    if total_billed == 0:
        return _answer(display_name, f"No billing history on file. Contact our billing department directly.")

    msg = (
        f"{display_name}, billing summary with {payer}: "
        f"total billed ${total_billed:,.2f}, covered ${total_covered:,.2f}, "
        f"your responsibility ${total_owed:,.2f}."
    )
    if denied_count > 0:
        msg += f" {denied_count} denied claim{'s' if denied_count > 1 else ''} may be adding to your balance — our billing team can review."
    msg += " For a procedure-specific estimate, call our financial counselors."

    return _answer(display_name, msg)


# ---------------------------------------------------------------------------
# 3. PRE-PROCEDURE PREP — instructions before a procedure
# ---------------------------------------------------------------------------

# Procedure prep instructions keyed by encounter type and description keywords
_PREP_INSTRUCTIONS = {
    "colonoscopy": (
        "For your colonoscopy, you must follow a clear liquid diet the day before. "
        "No solid food after midnight the night before your procedure. "
        "You will need to take a bowel prep solution as directed by your doctor — this is critical. "
        "Do not take blood thinners like aspirin or warfarin for 5 days before unless your doctor says otherwise. "
        "Arrange for someone to drive you home as you will be sedated."
    ),
    "surgery": (
        "For your surgery, do not eat or drink anything after midnight the night before. "
        "Stop blood thinners 5 days before unless your doctor instructs otherwise. "
        "Bring a list of all current medications to the hospital. "
        "Arrange for someone to drive you home. "
        "Arrive at least 2 hours before your scheduled procedure time for check-in and prep."
    ),
    "cardiac": (
        "For your cardiac procedure, take your regular medications with a small sip of water unless told otherwise. "
        "Do not eat or drink after midnight. "
        "Wear comfortable, loose clothing. "
        "Bring all your current medications and a list of any allergies. "
        "Someone must drive you home — you cannot drive yourself after sedation."
    ),
    "imaging": (
        "For your imaging scan, you may need to avoid eating for 4 hours beforehand depending on the type of scan. "
        "Remove all metal jewelry and piercings before arriving. "
        "If you have a pacemaker or metal implants, inform the technologist immediately. "
        "Wear comfortable clothing without metal fasteners."
    ),
    "default": (
        "For your upcoming procedure, please do not eat or drink anything after midnight the night before. "
        "Bring a photo ID, your insurance card, and a list of all current medications. "
        "Arrange for someone to drive you home if you will be receiving sedation or anesthesia. "
        "Arrive at least 30 minutes early for paperwork and check-in. "
        "If you have any questions about specific medications to stop, please call our nursing line."
    ),
}


def get_pre_procedure_prep(patient_name: str, graph) -> dict:
    """
    Use case: "Do I need to stop my medication?" / "When should I start fasting?"

    Looks up the patient's most recent upcoming or recent encounter type
    and returns appropriate prep instructions.
    """
    display_name, all_entities = _get_all_entities_for_person(patient_name, graph)
    if not display_name:
        return _not_found(patient_name)

    encounters = []
    for _, data in all_entities:
        attrs = data.get("attributes", {}) or {}
        if attrs.get("record_type") == "encounter":
            encounters.append(attrs)

    if not encounters:
        return _answer(
            display_name,
            f"{display_name}, I don't have a scheduled procedure on file for you. "
            f"Please call our scheduling department directly and they can provide specific preparation instructions for your procedure."
        )

    # Get the most recent encounter to determine procedure type
    latest = sorted(encounters, key=lambda e: e.get("encounter_date", ""), reverse=True)[0]
    description = (latest.get("description", "") + " " + latest.get("reason", "")).lower()
    enc_type    = latest.get("encounter_type", "").lower()
    date        = latest.get("encounter_date", "")

    # Match to prep instructions
    prep = _PREP_INSTRUCTIONS["default"]
    if any(kw in description for kw in ["colonoscopy", "colon", "endoscopy", "bowel"]):
        prep = _PREP_INSTRUCTIONS["colonoscopy"]
    elif any(kw in description for kw in ["surgery", "surgical", "operation", "arthroscopy", "fusion", "bypass"]):
        prep = _PREP_INSTRUCTIONS["surgery"]
    elif any(kw in description for kw in ["cardiac", "heart", "coronary", "catheter"]):
        prep = _PREP_INSTRUCTIONS["cardiac"]
    elif any(kw in description for kw in ["mri", "ct scan", "x-ray", "imaging", "scan", "ultrasound"]):
        prep = _PREP_INSTRUCTIONS["imaging"]
    elif "inpatient" in enc_type or "emergency" in enc_type:
        prep = _PREP_INSTRUCTIONS["surgery"]

    procedure_desc = latest.get("description", "your upcoming procedure")
    msg = f"{display_name}, for your {procedure_desc}"
    if date:
        msg += f" on {date}"
    msg += ": " + prep

    return _answer(display_name, msg)


# ---------------------------------------------------------------------------
# 4. BILL EXPLANATION — post-service bill dispute
# ---------------------------------------------------------------------------

def get_bill_explanation(patient_name: str, graph) -> dict:
    """
    Use case: "Why do I owe $1,500?" / "I thought this was covered!"

    Pulls the patient's billing history and explains the charges in plain English,
    including any denials that are contributing to the balance.
    """
    display_name, all_entities = _get_all_entities_for_person(patient_name, graph)
    if not display_name:
        return _not_found(patient_name)

    primary_attrs = {}
    encounters = []
    for _, data in all_entities:
        attrs = data.get("attributes", {}) or {}
        if "insurance_payer" in attrs:
            primary_attrs = attrs
        elif attrs.get("record_type") == "encounter":
            encounters.append(attrs)

    payer        = primary_attrs.get("insurance_payer", "your insurance")
    total_billed = float(primary_attrs.get("total_billed", 0))
    total_covered = float(primary_attrs.get("total_covered", 0))
    total_owed   = float(primary_attrs.get("total_owed", 0))
    denied_count = int(primary_attrs.get("denied_claims", 0))

    if total_owed == 0 and total_billed == 0:
        return _answer(display_name, f"No outstanding billing records found. Call billing with your bill reference number.")

    denied_encounters  = [e for e in encounters if e.get("claim_status") == "DENIED_NO_PA"]
    partial_encounters = [e for e in encounters if e.get("claim_status") == "PARTIAL"]

    msg = (
        f"{display_name}, your bill: ${total_billed:,.2f} billed, "
        f"{payer} paid ${total_covered:,.2f}, you owe ${total_owed:,.2f}. "
    )
    if denied_encounters:
        denied_total = sum(float(e.get("billed", 0)) for e in denied_encounters)
        msg += f"${denied_total:,.2f} from {len(denied_encounters)} denied claim{'s' if len(denied_encounters)>1 else ''} — prior auth not obtained. You can appeal these. "
    if partial_encounters:
        msg += f"Remaining balance is your deductible/coinsurance share under your {payer} plan. "
    if total_owed > 500:
        msg += f"We offer payment plans — contact billing to set one up."
    else:
        msg += (
            f"You can pay your balance of ${total_owed:,.2f} online at our patient portal, "
            f"by phone, or in person at the billing office."
        )

    return _answer(display_name, msg)
