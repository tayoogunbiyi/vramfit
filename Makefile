# GPU validation on a rented pod. Pass the pod's SSH address: make smoke HOST=1.2.3.4 SSH_PORT=22022
# (not PORT, which the validation scripts read as the vLLM port).
POD = bash validation/pod.sh

.PHONY: smoke eval fetch

# Ship, install, and run a short Qwen3-4B pass; results land in validation/runs/smoke/.
smoke:
	$(POD) ship $(HOST) $(SSH_PORT)
	$(POD) setup $(HOST) $(SSH_PORT)
	$(POD) smoke $(HOST) $(SSH_PORT)

# Full run for both models after confirmation (YES=1 skips it); results land in validation/runs/.
eval:
	$(POD) ship $(HOST) $(SSH_PORT)
	$(POD) setup $(HOST) $(SSH_PORT)
	YES=$(YES) $(POD) run $(HOST) $(SSH_PORT)

fetch:
	$(POD) fetch $(HOST) $(SSH_PORT)
