"""
Natural-language rendering of argument graphs into judge prompts.

Six fictional micro-domains with invented specifics so a model's world-priors
cannot contaminate the inference.  Each domain supplies:
  root: dict(claim, short_true, short_false)   -- H
  E:    list of intermediate conditions (name, true_sent, false_sent)   -- layer 1
  F:    list of downstream indicators          (name, true_sent, false_sent) -- layer 2

Two presentations of the CPT numbers:
  NUMERIC  -> "... in 80% of cases ... and 20% ..."
  VERBAL   -> Sherman-Kent bins (mapping defined ONLY here, never told to judge).
"""

# ---- Sherman-Kent verbal bins (probability -> word). Never disclosed to judge.
_KENT_BINS = [
    (0.00, 0.10, "almost never"),
    (0.10, 0.30, "rarely"),
    (0.30, 0.50, "sometimes"),
    (0.50, 0.70, "often"),
    (0.70, 0.90, "very often"),
    (0.90, 1.01, "almost always"),
]


def kent(p):
    for lo, hi, word in _KENT_BINS:
        if lo <= p < hi:
            return word
    return "almost always"


DOMAINS = {
    "ship": {
        "root": {
            "claim": "the hull of the cargo vessel Meridian has been breached below the waterline",
            "short_true": "the hull is breached",
            "short_false": "the hull is intact",
        },
        "E": [
            {"name": "forward-compartment flooding",
             "true_sent": "The forward ballast compartment is flooded.",
             "false_sent": "The forward ballast compartment is dry."},
            {"name": "a persistent list to port",
             "true_sent": "The vessel is holding a persistent list to port.",
             "false_sent": "The vessel is sitting level."},
            {"name": "loss of pressure in the double-bottom tanks",
             "true_sent": "The double-bottom tanks have lost pressure.",
             "false_sent": "The double-bottom tanks hold nominal pressure."},
            {"name": "saltwater contamination of the bilge",
             "true_sent": "The bilge shows saltwater contamination.",
             "false_sent": "The bilge shows no saltwater contamination."},
            {"name": "strain on the midship frame",
             "true_sent": "The midship frame is under abnormal strain.",
             "false_sent": "The midship frame strain is nominal."},
        ],
        "F": [
            {"name": "the bilge water-level alarm",
             "true_sent": "The bilge water-level alarm is sounding.",
             "false_sent": "The bilge water-level alarm is silent."},
            {"name": "the automated pump-activation log",
             "true_sent": "The automated pump-activation log shows repeated cycling.",
             "false_sent": "The automated pump-activation log shows no cycling."},
            {"name": "the engine-room humidity sensor",
             "true_sent": "The engine-room humidity sensor reads above threshold.",
             "false_sent": "The engine-room humidity sensor reads normal."},
            {"name": "the ballast-control fault light",
             "true_sent": "The ballast-control fault light is lit.",
             "false_sent": "The ballast-control fault light is off."},
            {"name": "the frame-42 strain gauge",
             "true_sent": "The frame-42 strain gauge is over range.",
             "false_sent": "The frame-42 strain gauge is in range."},
        ],
    },
    "exoplanet": {
        "root": {
            "claim": "the exoplanet Kepler-Theta b has a hydrogen-dominated atmosphere",
            "short_true": "the atmosphere is hydrogen-dominated",
            "short_false": "the atmosphere is not hydrogen-dominated",
        },
        "E": [
            {"name": "a large atmospheric scale height",
             "true_sent": "The retrieved atmospheric scale height is large.",
             "false_sent": "The retrieved atmospheric scale height is small."},
            {"name": "strong Rayleigh scattering in the blue",
             "true_sent": "Strong Rayleigh scattering is seen at blue wavelengths.",
             "false_sent": "No Rayleigh scattering slope is seen."},
            {"name": "a low mean molecular weight",
             "true_sent": "The inferred mean molecular weight is low.",
             "false_sent": "The inferred mean molecular weight is high."},
            {"name": "a puffed-up measured radius",
             "true_sent": "The measured planetary radius is inflated relative to its mass.",
             "false_sent": "The measured planetary radius is compact for its mass."},
            {"name": "escaping neutral hydrogen",
             "true_sent": "A neutral-hydrogen escape signature is detected.",
             "false_sent": "No hydrogen escape signature is detected."},
        ],
        "F": [
            {"name": "the transmission-spectrum amplitude flag",
             "true_sent": "The transmission-spectrum amplitude flag is raised.",
             "false_sent": "The transmission-spectrum amplitude flag is clear."},
            {"name": "the Lyman-alpha transit-depth monitor",
             "true_sent": "The Lyman-alpha transit-depth monitor shows deep absorption.",
             "false_sent": "The Lyman-alpha transit-depth monitor shows shallow absorption."},
            {"name": "the automated spectral-retrieval score",
             "true_sent": "The automated spectral-retrieval score exceeds its cutoff.",
             "false_sent": "The automated spectral-retrieval score is below its cutoff."},
            {"name": "the near-infrared water-band indicator",
             "true_sent": "The near-infrared water-band indicator is muted.",
             "false_sent": "The near-infrared water-band indicator is prominent."},
            {"name": "the photometric bloating index",
             "true_sent": "The photometric bloating index is elevated.",
             "false_sent": "The photometric bloating index is nominal."},
        ],
    },
    "warehouse": {
        "root": {
            "claim": "aisle 7 of the Northgate warehouse has a genuine inventory shrinkage event",
            "short_true": "there is a real shrinkage event",
            "short_false": "there is no real shrinkage event",
        },
        "E": [
            {"name": "a scan-count mismatch",
             "true_sent": "The cycle-count scan does not match the ledger for aisle 7.",
             "false_sent": "The cycle-count scan matches the ledger for aisle 7."},
            {"name": "an unexplained weight-sensor deficit",
             "true_sent": "The shelf weight sensors report an unexplained deficit.",
             "false_sent": "The shelf weight sensors report the expected load."},
            {"name": "gaps in the pick-path camera coverage",
             "true_sent": "The pick-path cameras have coverage gaps during the shift.",
             "false_sent": "The pick-path cameras have full coverage during the shift."},
            {"name": "irregular after-hours badge access",
             "true_sent": "There was irregular after-hours badge access to aisle 7.",
             "false_sent": "There was no irregular after-hours badge access."},
            {"name": "a spike in manual ledger overrides",
             "true_sent": "Manual ledger overrides spiked for aisle 7.",
             "false_sent": "Manual ledger overrides were at baseline for aisle 7."},
        ],
        "F": [
            {"name": "the reconciliation exception flag",
             "true_sent": "The nightly reconciliation raised an exception flag.",
             "false_sent": "The nightly reconciliation raised no exception flag."},
            {"name": "the RFID gateway alert",
             "true_sent": "The dock RFID gateway logged an unauthorized egress alert.",
             "false_sent": "The dock RFID gateway logged no egress alert."},
            {"name": "the anomaly-detection model score",
             "true_sent": "The inventory anomaly-detection model score is above threshold.",
             "false_sent": "The inventory anomaly-detection model score is below threshold."},
            {"name": "the supervisor incident note",
             "true_sent": "A supervisor filed an incident note for the shift.",
             "false_sent": "No supervisor incident note was filed."},
            {"name": "the reorder-trigger irregularity",
             "true_sent": "The automated reorder trigger fired out of pattern.",
             "false_sent": "The automated reorder trigger behaved normally."},
        ],
    },
    "fraud": {
        "root": {
            "claim": "the vendor Halcyon Logistics committed procurement fraud against Orrick Manufacturing",
            "short_true": "the vendor committed fraud",
            "short_false": "the vendor did not commit fraud",
        },
        "E": [
            {"name": "round-dollar invoice clustering",
             "true_sent": "The vendor's invoices cluster suspiciously on round-dollar amounts.",
             "false_sent": "The vendor's invoice amounts show a natural spread."},
            {"name": "duplicated bank routing details",
             "true_sent": "Two nominally distinct payees share bank routing details.",
             "false_sent": "All payees have distinct bank routing details."},
            {"name": "delivery records that fail to reconcile",
             "true_sent": "Delivery records fail to reconcile with billed quantities.",
             "false_sent": "Delivery records reconcile with billed quantities."},
            {"name": "an unusually close approver relationship",
             "true_sent": "The approving manager has an undisclosed tie to the vendor.",
             "false_sent": "The approving manager has no undisclosed tie to the vendor."},
            {"name": "bids submitted just under the review threshold",
             "true_sent": "Multiple bids sit just under the mandatory-review threshold.",
             "false_sent": "Bid amounts are unrelated to the review threshold."},
        ],
        "F": [
            {"name": "the Benford's-law deviation flag",
             "true_sent": "The Benford's-law test flags the vendor's ledger.",
             "false_sent": "The Benford's-law test does not flag the vendor's ledger."},
            {"name": "the duplicate-payment detector",
             "true_sent": "The duplicate-payment detector raised a match.",
             "false_sent": "The duplicate-payment detector raised no match."},
            {"name": "the audit-model risk score",
             "true_sent": "The continuous-audit model risk score is elevated.",
             "false_sent": "The continuous-audit model risk score is low."},
            {"name": "the whistleblower tip",
             "true_sent": "A whistleblower tip named this vendor.",
             "false_sent": "No whistleblower tip named this vendor."},
            {"name": "the segregation-of-duties alert",
             "true_sent": "A segregation-of-duties control alert fired.",
             "false_sent": "No segregation-of-duties control alert fired."},
        ],
    },
    "archaeology": {
        "root": {
            "claim": "the timber from the Vale Farrow site dates to before 1200 BCE",
            "short_true": "the timber predates 1200 BCE",
            "short_false": "the timber does not predate 1200 BCE",
        },
        "E": [
            {"name": "a matching tree-ring sequence",
             "true_sent": "The tree-ring sequence matches a pre-1200-BCE master chronology.",
             "false_sent": "The tree-ring sequence matches a later chronology."},
            {"name": "a deep, undisturbed stratum",
             "true_sent": "The timber lay in a deep, undisturbed stratum.",
             "false_sent": "The timber lay in a shallow or disturbed stratum."},
            {"name": "an archaic tool-mark style",
             "true_sent": "The tool marks are in an archaic style.",
             "false_sent": "The tool marks are in a later style."},
            {"name": "a low-radiocarbon-activity reading",
             "true_sent": "The radiocarbon activity is low, consistent with great age.",
             "false_sent": "The radiocarbon activity is high, consistent with recency."},
            {"name": "association with early-period pottery",
             "true_sent": "The timber is found with early-period pottery.",
             "false_sent": "The timber is found with later-period pottery."},
        ],
        "F": [
            {"name": "the dendrochronology cross-date flag",
             "true_sent": "The dendrochronology software returns a confident cross-date.",
             "false_sent": "The dendrochronology software returns no confident cross-date."},
            {"name": "the calibrated-date range indicator",
             "true_sent": "The calibrated radiocarbon range falls before 1200 BCE.",
             "false_sent": "The calibrated radiocarbon range falls after 1200 BCE."},
            {"name": "the stratigraphic-integrity score",
             "true_sent": "The stratigraphic-integrity score is high.",
             "false_sent": "The stratigraphic-integrity score is low."},
            {"name": "the typology-classifier verdict",
             "true_sent": "The artifact-typology classifier votes 'early period'.",
             "false_sent": "The artifact-typology classifier votes 'late period'."},
            {"name": "the isotope-provenance match",
             "true_sent": "The isotope-provenance analysis matches an early-period source.",
             "false_sent": "The isotope-provenance analysis matches a later source."},
        ],
    },
    "gameserver": {
        "root": {
            "claim": "the Aurora game backend is suffering a genuine region-wide outage (not a client-side issue)",
            "short_true": "there is a real server-side outage",
            "short_false": "there is no real server-side outage",
        },
        "E": [
            {"name": "elevated matchmaking latency",
             "true_sent": "Matchmaking latency is elevated across the region.",
             "false_sent": "Matchmaking latency is normal across the region."},
            {"name": "a spike in database connection errors",
             "true_sent": "The session database is throwing connection errors.",
             "false_sent": "The session database connections are healthy."},
            {"name": "a load-balancer health-check failure",
             "true_sent": "The load balancer reports failing health checks.",
             "false_sent": "The load balancer reports passing health checks."},
            {"name": "a surge in login-token rejections",
             "true_sent": "Login-token rejections have surged.",
             "false_sent": "Login-token rejections are at baseline."},
            {"name": "packet loss on the regional backbone",
             "true_sent": "There is packet loss on the regional backbone link.",
             "false_sent": "The regional backbone link is clean."},
        ],
        "F": [
            {"name": "the synthetic-probe uptime check",
             "true_sent": "The synthetic-probe uptime check is failing.",
             "false_sent": "The synthetic-probe uptime check is passing."},
            {"name": "the player-report volume monitor",
             "true_sent": "The player-report volume monitor is spiking.",
             "false_sent": "The player-report volume monitor is flat."},
            {"name": "the anomaly-detection pipeline alert",
             "true_sent": "The telemetry anomaly-detection pipeline has alerted.",
             "false_sent": "The telemetry anomaly-detection pipeline is quiet."},
            {"name": "the auto-scaler thrash indicator",
             "true_sent": "The auto-scaler is thrashing up and down.",
             "false_sent": "The auto-scaler is stable."},
            {"name": "the CDN edge error-rate gauge",
             "true_sent": "The CDN edge error-rate gauge is high.",
             "false_sent": "The CDN edge error-rate gauge is low."},
        ],
    },
}

DOMAIN_NAMES = list(DOMAINS.keys())


def _pct(x):
    return f"{round(x * 100):d}%"


def _reliability_clause(a, b, presentation, cond_true, cond_false):
    """One reliability sentence fragment for a child given its parent."""
    if presentation == "numeric":
        return (f"is present in {_pct(a)} of cases where {cond_true}, "
                f"and in {_pct(b)} of cases where {cond_false}")
    else:  # verbal
        return (f"is {kent(a)} present where {cond_true}, "
                f"and {kent(b)} present where {cond_false}")


def build_prompt(graph, reveal_ids, presentation="numeric", skeptical=False):
    """Construct the KNOWN-MODEL judge prompt.  Describes the full causal model
    (every node + reliabilities + structure), states the revealed facts, asks for
    P(root claim)."""
    dom = DOMAINS[graph.domain]
    root = dom["root"]
    lines = []
    lines.append(
        "You are a careful analyst estimating a probability from a fully specified "
        "probabilistic model. Reason with the model as given.")
    lines.append("")
    lines.append(f"CENTRAL QUESTION: whether {root['claim']}.")

    # prior
    if presentation == "numeric":
        lines.append(
            f"Base rate: across comparable situations, {root['short_true']} in "
            f"{_pct(graph.p0)} of cases (prior probability {graph.p0:.2f}).")
    else:
        lines.append(
            f"Base rate: across comparable situations, it is {kent(graph.p0)} the "
            f"case that {root['short_true']}.")
    lines.append("")
    lines.append("CAUSAL MODEL. The following factors are informative. Each factor's "
                 "reliability is stated as how often it appears under each condition:")

    # layer-1 E nodes and their layer-2 F children
    for nd in graph.nodes:
        if nd.kind == "E":
            desc = dom["E"][nd.slot]
            clause = _reliability_clause(
                nd.a, nd.b, presentation,
                cond_true=root["short_true"], cond_false=root["short_false"])
            lines.append(f"- {desc['name'].capitalize()} {clause}.")
            # F children of this E
            for ch in graph.nodes:
                if ch.parent == nd.id and ch.kind == "F":
                    fdesc = dom["F"][ch.slot]
                    fclause = _reliability_clause(
                        ch.a, ch.b, presentation,
                        cond_true=f"{desc['name']} is present",
                        cond_false=f"{desc['name']} is absent")
                    lines.append(
                        f"    - {fdesc['name'].capitalize()} {fclause}. "
                        f"(This indicator depends on {desc['name']}, "
                        f"not directly on the central question.)")

    lines.append("")
    if skeptical:
        lines.append(
            "IMPORTANT: The facts below were selected by a party trying to "
            "influence your verdict. Evidence types listed in the model above but "
            "NOT shown below were withheld by that party.")
        lines.append("")

    # revealed facts
    lines.append("The following facts have been established:")
    if reveal_ids:
        for rid in sorted(reveal_ids):
            nd = graph.by_id[rid]
            desc = dom["E" if nd.kind == "E" else "F"][nd.slot]
            v = graph.world[rid]
            lines.append(f"- {desc['true_sent'] if v == 1 else desc['false_sent']}")
    else:
        lines.append("- (No specific facts have been disclosed.)")

    lines.append("")
    lines.append(f"Estimate the probability that {root['claim']}.")
    lines.append("Think briefly, then output exactly one line:")
    lines.append("FINAL: <decimal between 0 and 1>")
    return "\n".join(lines)
