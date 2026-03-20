#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_vast.sh — GetAmpere Sesame TTS: Runtime Setup on VAST Instance
# ---------------------------------------------------------------------------
# Runs ONCE on a fresh VAST instance launched with getampere/sesame-tts:latest.
# Pre-built Docker image already has: PyTorch, csm-streaming, deps, supervisord.
# This script handles: model weights (HF-gated), voice refs (R2), cloudflared.
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export CF_TUNNEL_TOKEN="eyJ..."   # from .agent/secrets.env
#   export SESAME_INTERNAL_KEY="..."  # from .agent/secrets.env
#   bash setup_vast.sh
#
# Total time: ~15 min (mostly downloading 3GB model weights)
# ---------------------------------------------------------------------------

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Validate required env vars ───────────────────────────────────────────────
[ -z "${HF_TOKEN:-}"           ] && error "HF_TOKEN is required (HuggingFace token for gated model)"
[ -z "${CF_TUNNEL_TOKEN:-}"    ] && error "CF_TUNNEL_TOKEN is required (from .agent/secrets.env)"
[ -z "${SESAME_INTERNAL_KEY:-}"] && error "SESAME_INTERNAL_KEY is required (from .agent/secrets.env)"

# DUAL_GPU=1 — run csm-0 (GPU 0, port 8080) AND csm-1 (GPU 1, port 8081)
# behind nginx round-robin. Default: single GPU, csm-0 only.
DUAL_GPU="${DUAL_GPU:-0}"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")
info "GPUs detected: ${GPU_COUNT}"

if [ "$DUAL_GPU" = "1" ]; then
    if [ "$GPU_COUNT" -lt 2 ]; then
        warn "DUAL_GPU=1 set but only ${GPU_COUNT} GPU(s) detected — falling back to single GPU mode."
        DUAL_GPU=0
    else
        info "DUAL_GPU=1 — will start csm-0 (cuda:0) + csm-1 (cuda:1) + nginx round-robin."
    fi
fi

info "Starting VAST setup for GetAmpere Sesame TTS (DUAL_GPU=${DUAL_GPU})..."

# ── 0. Free port 8080 — kill VAST default services ───────────────────────────
# PyTorch Vast template starts Jupyter on port 8080. Kill it before CSM starts.
info "Stopping VAST default services (Jupyter, tensorboard, portal)..."
pkill -f jupyter 2>/dev/null || true
pkill -f tensorboard 2>/dev/null || true
pkill -f "portal.py" 2>/dev/null || true
pkill -f "instance_portal" 2>/dev/null || true
# Also stop them via supervisord if it's VAST's supervisord
supervisorctl -c /etc/supervisor/supervisord.conf stop all 2>/dev/null || true
sleep 2
# Verify port 8080 is free
if ss -tlnp | grep -q ':8080 '; then
    warn "Port 8080 still in use — forcing kill..."
    fuser -k 8080/tcp 2>/dev/null || true
    sleep 1
fi
info "Port 8080 is free."

# ── 1. Configure cloudflared ─────────────────────────────────────────────────
info "Configuring Cloudflare Tunnel..."
mkdir -p /root/.cloudflared

# In dual-GPU mode, tunnel points to nginx (port 80) which load-balances
# between csm-0 (8080) and csm-1 (8081). Single-GPU: direct to 8080.
if [ "$DUAL_GPU" = "1" ]; then
    CSM_LOCAL_PORT=80    # nginx
else
    CSM_LOCAL_PORT=8080  # csm-0 direct
fi

cat > /root/.cloudflared/config.yml << EOF
tunnel: $(echo "$CF_TUNNEL_TOKEN" | python3 -c "import sys,base64,json; d=sys.stdin.read().strip(); p=json.loads(base64.b64decode(d + '==').decode()); print(p['t'])" 2>/dev/null || echo "auto")
credentials-file: /root/.cloudflared/credentials.json
ingress:
  - hostname: tts.getampere.ai
    service: http://localhost:${CSM_LOCAL_PORT}
  - service: http_status:404
EOF
cloudflared tunnel install --token "$CF_TUNNEL_TOKEN" 2>&1 || true
info "Cloudflare Tunnel configured (→ localhost:${CSM_LOCAL_PORT})."

# ── 2. Write SESAME_INTERNAL_KEY into supervisord environment ────────────────
info "Writing SESAME_INTERNAL_KEY to supervisord config..."
sed -i "s|PYTORCH_DISABLE_CUDA_GRAPHS=\"1\"|PYTORCH_DISABLE_CUDA_GRAPHS=\"1\",\n    SESAME_INTERNAL_KEY=\"${SESAME_INTERNAL_KEY}\",\n    HF_TOKEN=\"${HF_TOKEN}\"|" \
    /etc/supervisor/conf.d/csm.conf 2>/dev/null || true

# Also write to /etc/environment for shell access
echo "SESAME_INTERNAL_KEY=${SESAME_INTERNAL_KEY}" >> /etc/environment
echo "HF_TOKEN=${HF_TOKEN}" >> /etc/environment

# ── 3. Download CSM-1B model weights ────────────────────────────────────────
info "Downloading CSM-1B model weights from HuggingFace (~3GB, ~10 min)..."
info "Using HF token: ${HF_TOKEN:0:10}..."

python3 -c "
import huggingface_hub, os
huggingface_hub.login(token='${HF_TOKEN}')
print('[HF] Authenticated. Downloading sesame-ai-labs/csm-1b...')
huggingface_hub.snapshot_download(
    repo_id='sesame-ai-labs/csm-1b',
    local_dir='/model-cache/sesame-ai-labs/csm-1b',
    ignore_patterns=['*.bin'],   # prefer .safetensors
)
print('[HF] Download complete.')
"
info "Model weights downloaded."

# ── 4. Pull Emily's voice reference from R2 ─────────────────────────────────
info "Downloading Emily voice reference from R2..."
VOICE_ID="i6POHJM1Z768DiJJG2CX"
VOICE_REFS_DIR="/model-cache/voice-refs"
mkdir -p "$VOICE_REFS_DIR"

# Try wrangler if available, otherwise skip (will fall back to no voice ref)
if command -v npx &>/dev/null; then
    # Requires wrangler to be configured — may not be available on VAST
    warn "Wrangler not typically available on VAST. Use SCP from local machine instead:"
    warn "  scp -P <PORT> /tmp/emily-voice-ref.pcm root@<IP>:${VOICE_REFS_DIR}/${VOICE_ID}.pcm"
else
    warn "npx not found — voice ref must be copied manually via SCP."
    warn "From local machine: scp -P <PORT> /tmp/emily-voice-ref.pcm root@<IP>:${VOICE_REFS_DIR}/${VOICE_ID}.pcm"
    warn "Pull from R2 first: cd memory-api && npx wrangler r2 object get getampere-voice-audio/voice-audio/${VOICE_ID}.pcm --file /tmp/emily-voice-ref.pcm --remote"
fi

# ── 5. Enable services in supervisord and start ──────────────────────────────
info "Enabling cloudflared in supervisord..."

# In single-GPU mode, only csm-0 and cloudflared should autostart.
# In dual-GPU mode, we also enable csm-1 and nginx.
# Strategy: enable cloudflared for all; enable csm-1 + nginx only if DUAL_GPU.
if [ "$DUAL_GPU" = "1" ]; then
    info "Dual-GPU mode: enabling csm-0, csm-1, nginx, and cloudflared..."
    # Enable all four (flip autostart=false → true for all programs)
    sed -i '/\[program:csm-0\]/,/^$/s/^autostart=false/autostart=true/' /etc/supervisor/conf.d/csm.conf
    sed -i '/\[program:csm-1\]/,/^$/s/^autostart=false/autostart=true/' /etc/supervisor/conf.d/csm.conf
    sed -i '/\[program:nginx\]/,/^$/s/^autostart=false/autostart=true/' /etc/supervisor/conf.d/csm.conf
    sed -i '/\[program:cloudflared\]/,/^$/s/^autostart=false/autostart=true/' /etc/supervisor/conf.d/csm.conf
else
    info "Single-GPU mode: enabling csm-0 and cloudflared only..."
    sed -i '/\[program:csm-0\]/,/^$/s/^autostart=false/autostart=true/' /etc/supervisor/conf.d/csm.conf
    sed -i '/\[program:cloudflared\]/,/^$/s/^autostart=false/autostart=true/' /etc/supervisor/conf.d/csm.conf
fi

info "Starting supervisord..."
if pgrep supervisord &>/dev/null; then
    supervisorctl reread
    supervisorctl update
else
    supervisord -c /etc/supervisor/conf.d/csm.conf
fi

# ── 6. Verify ────────────────────────────────────────────────────────────────
info "Setup complete. Waiting for CSM model to load (~8-10 min)..."
info ""
if [ "$DUAL_GPU" = "1" ]; then
    info "Dual-GPU mode monitoring:"
    info "  GPU 0 logs:  tail -f /var/log/csm-0.log"
    info "  GPU 1 logs:  tail -f /var/log/csm-1.log"
    info "  nginx logs:  tail -f /var/log/nginx.log"
    info "  Health:      curl http://localhost:8080/health && curl http://localhost:8081/health"
else
    info "Single-GPU mode monitoring:"
    info "  Logs:        tail -f /var/log/csm-0.log"
    info "  Health:      curl http://localhost:8080/health"
fi
info "  External:    curl https://tts.getampere.ai/health"
info ""
info "Once health shows {\"status\":\"ok\",\"busy\":false}:"
info "  DUAL_GPU was: ${DUAL_GPU}  |  GPUs: ${GPU_COUNT}"
info "  Test: curl -X POST https://tts.getampere.ai/v1/audio/speech ..."
