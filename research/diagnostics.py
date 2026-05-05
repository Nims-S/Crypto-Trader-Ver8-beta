def build_candidate_diagnostics(payload: dict) -> dict:
    wf = (payload or {}).get("walk_forward") or {}
    return {
        "trade_activity": {},
        "top_fail_reasons": wf.get("reasons") or [],
        "score_spread": wf.get("score_spread", 0.0),
    }