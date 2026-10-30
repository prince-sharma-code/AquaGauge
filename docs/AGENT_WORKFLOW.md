# Shared-agent operating rules

1. Use one Git branch per workstream. Never let two agents edit the same files concurrently.
2. Give an agent one deliverable with a file allowlist, API contract, and acceptance tests.
3. Ask for a plan first; approve the plan; then allow edits. Avoid vague prompts like “build the whole platform.”
4. Require every agent to run its own tests/build and report the exact results. Review diffs rather than trusting summaries.
5. Merge small commits frequently. One person owns main and integration.
6. Run agents in parallel only when files/interfaces are independent. If two agents propose competing architectures, choose one immediately; do not spend an hour comparing.
7. No new packages after hour 16 unless they fix a blocker.
8. Keep API keys/secrets out of prompts, terminal recordings, `.env` commits, and the repository.
9. Do not claim “real-time”, “accurate to centimetres”, “production-ready”, “privacy compliant”, or “automatic rerouting” unless tested and qualified. Call simulated outputs simulated.
