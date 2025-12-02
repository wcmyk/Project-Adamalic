# AngelOS System Blueprint

This document distills the system design for AngelOS, a distributed multi-agent AI operating system. It summarizes the planes, key components, and the recommended first build target.

## System Objectives
- **Control Plane:** Dashboard, Supervisor Angel (ADAM), auth/RBAC, and policy enforcement.
- **Data Plane:** CRDT-based AngelGit storage (LELIEL/IROUEL), relational and cache backends, metrics/logs, and agent memory (ARMISAEL).
- **Execution Plane:** Isolated runtimes (SHAMSHEL, SACHIEL, MATARAEL, RAMIEL, BARDIEL) with sandboxing, quotas, and language-specific containers.
- **Communication Plane:** WebSocket mesh (ARAEL) with message router, mTLS, ACK/Retry, and routing tables.
- **ML Plane:** Tiny role-specific models (LILITH) with LoRA adapters, vector memory, registry, and guarded fine-tuning pipelines.
- **Governance:** Supervisor oversight, approvals, rollback, policy checks, rogue-angel detection, and immutable logging.

## Distributed Architecture
- **VPS layout:** Control Plane + Dashboard + Supervisor; dedicated storage node (Postgres + Redis); multiple execution nodes; labs node for canaries; network/API gateway node for the WebSocket hub.
- **Transport:** TLS-encrypted WebSockets with mutual TLS for node identity; JSON/protobuf envelopes with task/event/log/heartbeat types; backpressure-aware retry queues.
- **Versioning:** CRDT filesystem (Automerge/Yjs style) backed by Redis Streams for mutation logs and Postgres for snapshots and rollback; Supervisor gates major commits.

## Execution & Security
- **Sandboxes:** One container per language or agent with read-only roots, disabled network, non-root users, seccomp/AppArmor profiles, CPU/memory quotas, and watchdog timeouts.
- **Languages:** Python 3.11, Node.js/TS, Rust, Go, .NET, Swift; Python guarded with AST sanitization, disallowed imports, and infinite-loop detection.
- **Risk controls:** Rogue-angel heuristics (commit frequency, syscall anomalies, CPU spikes, self-referential code), quarantining, and auto-restart health checks.

## CI/CD & Testing
- Patch → CRDT merge to dev → automated tests (unit/integration/security/static) → Supervisor review → merge to main → execution nodes auto-pull.
- Log and metric streaming to the dashboard; append-only logs and snapshot compaction to mitigate CRDT blow-up.

## Recommended First Build
Start with the **Foundation (Phase 1)** to unblock all other work:
1. **Control Plane skeleton:** Basic dashboard shell, auth/RBAC, and Supervisor Angel stub with policy hooks.
2. **Communication hub:** WebSocket router (ARAEL) with mTLS, routing table, heartbeat/ACK, and retry queues.
3. **Storage baseline:** Postgres + Redis provisioning plus minimal CRDT log/snapshot services (LELIEL/IROUEL) and immutable logging.

These pieces establish secure identity, routing, and persistence so subsequent execution sandboxes, AngelGit integration, and ML adapters can be layered safely.

## LLM Reference Implementation
The `llm` package contains a minimal GPT-style decoder suitable for small-scale character modeling experiments and integration tests:
- **Architecture:** Transformer encoder stack used as a causal decoder with learned token + positional embeddings, GELU activations, dropout, and tied projection head (`llm/model.py`).
- **Configuration:** Dataclass-driven model/training hyperparameters to keep runs reproducible (`llm/config.py`).
- **Data pipeline:** Deterministic character tokenizer and windowed next-token dataset for quick corpora bootstrapping (`llm/data.py`).
- **Training loop:** Warmup+cosine LR schedule, gradient clipping, AdamW optimizer, and optional checkpoint emission (`llm/train.py`).

Example sketch:
```python
from llm import CharacterTokenizer, GPTDecoder, ModelConfig, TrainingConfig, train

corpus = ["angelos is online\n"]
tokenizer = CharacterTokenizer(corpus)
model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
train_config = TrainingConfig(max_steps=500, device="cpu")
model = train(corpus, model_config, train_config, block_size=64, checkpoint_path="artifacts/llm.pt")
```
