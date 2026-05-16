#!/bin/bash
set -euo pipefail

cat <<'EOF'
start_dpo.sh is archived and no longer supported in the active training runtime.

Why:
- legacy DPO scripts were moved under experiments/legacy_post_training/
- mainline training support is currently SFT-first under src/eliza_trainer/sft/

Use instead:
1) SFT runs via ./start_sft.sh <config>
2) If needed, run legacy DPO manually from experiments/legacy_post_training/ at your own risk

Planned path:
- new TRL-native DPO runner under src/eliza_trainer/dpo/
EOF

exit 1
