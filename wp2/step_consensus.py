"""WP2 step: validate raw extractions, build consensus claims, compute agreement (Paper 1, RQ2).

Outputs under results/real_v2/wp2/: extraction/validated.jsonl.gz, claims/consensus.jsonl,
agreement.json (per-model rates, pairwise F1, Krippendorff alpha with bootstrap CI, regex
recall, extraction flow counts).

Usage:
    python wp2_consistency.py consensus
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from typing import Any, Optional

from . import common as C
from . import consensus as K

logger = logging.getLogger("wp2.step_consensus")


def run(cfg: dict[str, Any], only: Optional[str] = None, force: bool = False) -> int:
    res_dir = C.results_dir(cfg)
    panel = K.final_panel(cfg, res_dir)
    rows, per_model = K.validate_all(cfg)
    with gzip.open(res_dir / "extraction" / "validated.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    pre = K.build_units(rows, panel)
    merged, minfo = K.merge_units(pre)
    cons = K.consensus_rows(merged, panel)
    C.write_jsonl(res_dir / "claims" / "consensus.jsonl", cons)

    min_votes = 2
    by_type = Counter(r["tcode"] for r in cons if r["votes"] >= min_votes)
    flow = {
        "models": panel,
        "units_union_pre_merge": len(pre),
        "units_after_path_merge": len(merged),
        "path_merges": minfo["merged_units"],
        "votes_distribution_after_merge": dict(Counter(r["votes"] for r in cons)),
        "consensus_ge2": sum(1 for r in cons if r["votes"] >= min_votes),
        "consensus_ge2_by_type": dict(by_type),
        "consensus_ge2_mode": dict(Counter(r["mode"] for r in cons if r["votes"] >= min_votes)),
        "consensus_ge2_asserts_existing": sum(1 for r in cons if r["votes"] >= min_votes and r["mode"] == "asserts_existing"),
        "union_asserts_existing": sum(1 for r in cons if r["mode"] == "asserts_existing"),
        "consensus3_asserts_existing": sum(1 for r in cons if r["votes"] == 3 and r["mode"] == "asserts_existing"),
        "unanimous_mode_asserts_ge2": sum(1 for r in cons if r["votes"] >= 2 and r["mode"] == "asserts_existing"
                                          and r["mode_unanimous"]),
        "protected_status_ge2": sum(1 for r in cons if r["votes"] >= min_votes and r["tcode"] == "T2"
                                    and r["attrs"].get("status_kind") == "protected"),
    }
    ag = K.agreement_report(cfg, rows, pre, panel)
    ag["per_model"] = per_model
    rows_s, pm_s = K.validate_all(cfg, repair=False)
    pre_s = K.build_units(rows_s, panel)
    ag["strict_no_quote_repair"] = {
        "note": "same pipeline but claims without a model-supplied quote are rejected instead of repaired from the subject's line",
        "per_model_accepted": {m: v.get("accepted_units_raw", 0) for m, v in pm_s.items()},
        "agreement": K.agreement_report(cfg, rows_s, pre_s, panel)}
    ag["flow"] = flow
    ag["regex_recall"] = K.regex_recall(cfg, cons, pre)
    C.write_json(res_dir / "agreement.json", ag)
    logger.info("consensus: %d units union, %d with >=2 votes, %d asserting; alpha=%.3f F1=%.3f",
                len(pre), flow["consensus_ge2"], flow["consensus_ge2_asserts_existing"],
                ag["alpha_presence"]["alpha"], ag["pairwise_f1_mean"])
    return 0
