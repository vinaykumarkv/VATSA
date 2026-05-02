import json
from datetime import datetime

class AuditLogger:
    def __init__(self, log_path="logs/vatsa_audit.jsonl"):
        self.log_path = log_path
    
    def record(self, modalities_used, fused_embedding_hash, 
               action_taken, confidence, timestamp=None):
        entry = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "modalities": modalities_used,
            "embedding_hash": fused_embedding_hash,
            "action": action_taken,
            "confidence": confidence
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')