# config.py — add this now, wire it up later
VATSA_CONFIG = {
    "human_oversight_required": True,    # action head always defers
    "max_action_tier": 1,                # only safe/log actions allowed
    "audit_logging": True,               # log all inferences
    "pii_filter": False,                 # not implemented yet — placeholder
    "confidence_threshold": 0.95         # minimum confidence to act
}