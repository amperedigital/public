#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_vast.sh — GetAmpere Sesame TTS: Runtime Setup on VAST Instance
# ---------------------------------------------------------------------------
# Runs ONCE on a fresh VAST instance launched with ghcr.io/amperedigital/public:sesame-tts
# Pre-built Docker image already has: PyTorch, csm-streaming, deps, supervisord.
# This script handles: model weights (HF-gated), voice refs, cloudflared.
#
# Usage:
#   export HF_TOKEN="hf_..."
#   export CF_TUNNEL_TOKEN="eyJ..."   # from .agent/secrets.env
#   export SESAME_INTERNAL_KEY="..."  # from .agent/secrets.env
#   bash setup_vast.sh
#
# Total time: ~15 min (mostly downloading 3GB model weights)
#
# Provisioning: bash /home/drewman/getampere-ag-build/sesame-tts/provision_vast.sh
# ---------------------------------------------------------------------------

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Validate required env vars ───────────────────────────────────────────────
[ -z "${HF_TOKEN:-}"           ] && error "HF_TOKEN is required"
[ -z "${CF_TUNNEL_TOKEN:-}"    ] && error "CF_TUNNEL_TOKEN is required"
[ -z "${SESAME_INTERNAL_KEY:-}"] && error "SESAME_INTERNAL_KEY is required"

# DUAL_GPU=1 — run csm-0 (GPU 0) + csm-1 (GPU 1) behind nginx. Default: single GPU.
DUAL_GPU="${DUAL_GPU:-0}"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "1")
info "GPUs detected: ${GPU_COUNT}"

if [ "$DUAL_GPU" = "1" ] && [ "$GPU_COUNT" -lt 2 ]; then
    warn "DUAL_GPU=1 but only ${GPU_COUNT} GPU(s) — falling back to single GPU."
    DUAL_GPU=0
fi

info "Starting VAST setup for GetAmpere Sesame TTS (DUAL_GPU=${DUAL_GPU})..."

# ── 0. Triton/CUDA linker fix ────────────────────────────────────────────────
# PyTorch runtime images lack libcuda.so (dev symlink) — Triton needs it to link
# its CUDA driver utilities at startup. Without this, the server crashes in a loop.
info "Ensuring libcuda.so dev symlink exists for Triton..."
LIBCUDA=$(find /usr/lib -name "libcuda.so.1" 2>/dev/null | head -1)
if [ -n "$LIBCUDA" ]; then
    LIBCUDA_DIR=$(dirname "$LIBCUDA")
    ln -sf "$LIBCUDA" "${LIBCUDA_DIR}/libcuda.so" 2>/dev/null || true
    info "Symlink: ${LIBCUDA_DIR}/libcuda.so → libcuda.so.1"
fi

# ── 1. Free port 8080 ────────────────────────────────────────────────────────
info "Stopping VAST default services..."
pkill -f jupyter 2>/dev/null || true
pkill -f tensorboard 2>/dev/null || true
pkill -f "portal.py" 2>/dev/null || true
pkill -f "instance_portal" 2>/dev/null || true
supervisorctl -c /etc/supervisor/supervisord.conf stop all 2>/dev/null || true
sleep 2
if ss -tlnp | grep -q ':8080 '; then
    warn "Port 8080 still in use — forcing kill..."
    fuser -k 8080/tcp 2>/dev/null || true
    sleep 1
fi
info "Port 8080 is free."

# ── 2. Configure cloudflared (token mode — no credentials.json needed) ──────
info "Configuring Cloudflare Tunnel (token mode)..."

if [ "$DUAL_GPU" = "1" ]; then
    CSM_LOCAL_PORT=80   # nginx
else
    CSM_LOCAL_PORT=8080 # csm-0 direct
fi

# Inject CF_TUNNEL_TOKEN into the cloudflared supervisord command
# supervisord.conf ships with: command=... cloudflared tunnel --no-autoupdate run
# We add: --token <TOKEN>
sed -i "s|command=/usr/local/bin/cloudflared tunnel --no-autoupdate run$|command=/usr/local/bin/cloudflared tunnel --no-autoupdate run --token ${CF_TUNNEL_TOKEN}|" \
    /etc/supervisor/conf.d/csm.conf 2>/dev/null || true

info "Cloudflare Tunnel configured (→ localhost:${CSM_LOCAL_PORT})."

# ── 3. Write runtime secrets into supervisord environment ────────────────────
info "Writing runtime secrets to supervisord config..."
sed -i "s|PYTORCH_DISABLE_CUDA_GRAPHS=\"1\"|PYTORCH_DISABLE_CUDA_GRAPHS=\"1\",\n    SESAME_INTERNAL_KEY=\"${SESAME_INTERNAL_KEY}\",\n    HF_TOKEN=\"${HF_TOKEN}\"|" \
    /etc/supervisor/conf.d/csm.conf 2>/dev/null || true

echo "SESAME_INTERNAL_KEY=${SESAME_INTERNAL_KEY}" >> /etc/environment
echo "HF_TOKEN=${HF_TOKEN}" >> /etc/environment

# ── 4. Download CSM-1B model weights ────────────────────────────────────────
# Prefer R2 (fast, datacenter speeds, ~2 min) over HuggingFace (~15 min, gated).
# R2 credentials from secrets.env: R2_ACCESS_KEY_ID + R2_SECRET_ACCESS_KEY
R2_ENDPOINT="https://fdc6cebcbbf7c7b33cc6a0e59ac8cd5f.r2.cloudflarestorage.com"
R2_BUCKET="getampere-model-weights"
MODEL_CACHE="/model-cache/hub/models--sesame--csm-1b"

if [ -d "$MODEL_CACHE/snapshots" ]; then
    info "Model weights already on disk — skipping download."
elif [ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_SECRET_ACCESS_KEY:-}" ]; then
    info "Downloading CSM-1B weights from R2 (~19GB, ~2-4 min at datacenter speeds)..."
    pip install awscli -q 2>/dev/null
    aws configure set aws_access_key_id "${R2_ACCESS_KEY_ID}"
    aws configure set aws_secret_access_key "${R2_SECRET_ACCESS_KEY}"
    aws configure set default.region auto
    aws s3 sync \
        "s3://${R2_BUCKET}/hub/models--sesame--csm-1b/" \
        "$MODEL_CACHE/" \
        --endpoint-url "$R2_ENDPOINT" \
        --no-progress \
    && info "Model weights downloaded from R2." \
    || { warn "R2 download failed — falling back to HuggingFace..."; FALLBACK_HF=1; }
else
    warn "R2 credentials not set — falling back to HuggingFace (~15 min)..."
    FALLBACK_HF=1
fi

if [ "${FALLBACK_HF:-0}" = "1" ]; then
    info "Downloading CSM-1B model weights from HuggingFace (~3GB download, ~15 min)..."
    info "HF token: ${HF_TOKEN:0:10}..."
    python3 -c "
import huggingface_hub
huggingface_hub.login(token='${HF_TOKEN}')
print('[HF] Authenticated. Downloading sesame/csm-1b to HF cache...')
huggingface_hub.snapshot_download(
    repo_id='sesame/csm-1b',
    cache_dir='/model-cache/hub',
    ignore_patterns=['*.bin'],
)
print('[HF] Download complete.')
"
    info "Model weights downloaded from HuggingFace."
fi


# ── 5. Pull voice references ─────────────────────────────────────────────────
# Voice refs live on the server permanently for zero-latency inference.
# They are pulled from the public GitHub repo at startup.
# R2 stores the canonical copies (getampere-voice-audio bucket).
info "Downloading voice references..."
VOICE_REFS_DIR="/model-cache/voice-refs"
mkdir -p "$VOICE_REFS_DIR"
BASE_VOICE_URL="https://raw.githubusercontent.com/amperedigital/public/main/sesame-tts/voice-refs"

declare -A VOICES=(
    ["i6POHJM1Z768DiJJG2CX"]="emily"
)

for VOICE_ID in "${!VOICES[@]}"; do
    NAME="${VOICES[$VOICE_ID]}"
    OUT="${VOICE_REFS_DIR}/${VOICE_ID}.pcm"
    if [ -f "$OUT" ]; then
        info "Voice ref already exists: ${NAME} (${VOICE_ID})"
        continue
    fi
    info "Downloading voice ref: ${NAME} (${VOICE_ID})..."
    if curl -fsSL --max-time 30 "${BASE_VOICE_URL}/${VOICE_ID}.pcm" -o "$OUT" 2>/dev/null; then
        info "Voice ref downloaded: ${NAME} — $(du -h $OUT | cut -f1)"
    else
        warn "Could not download voice ref ${NAME} — will run without it."
    fi
done

# ── 6. Enable services and start supervisord ─────────────────────────────────
info "Enabling services in supervisord..."
if [ "$DUAL_GPU" = "1" ]; then
    info "Dual-GPU mode: enabling csm-0, csm-1, nginx, cloudflared..."
    for program in csm-0 csm-1 nginx cloudflared; do
        sed -i "/\[program:${program}\]/,/^$/s/^autostart=false/autostart=true/" \
            /etc/supervisor/conf.d/csm.conf
    done
else
    info "Single-GPU mode: enabling csm-0 and cloudflared only..."
    for program in csm-0 cloudflared; do
        sed -i "/\[program:${program}\]/,/^$/s/^autostart=false/autostart=true/" \
            /etc/supervisor/conf.d/csm.conf
    done
fi

info "Starting supervisord..."
if pgrep supervisord >/dev/null; then
    supervisorctl reread
    supervisorctl update
else
    supervisord -c /etc/supervisor/conf.d/csm.conf
fi

# ── 7. Done ──────────────────────────────────────────────────────────────────
info "Setup complete. Waiting for CSM model to load (~5-8 min)..."
info ""
if [ "$DUAL_GPU" = "1" ]; then
    info "Dual-GPU monitoring:"
    info "  GPU 0 logs:  tail -f /var/log/csm-0.log"
    info "  GPU 1 logs:  tail -f /var/log/csm-1.log"
    info "  Health:      curl http://localhost:8080/health && curl http://localhost:8081/health"
else
    info "Single-GPU monitoring:"
    info "  Logs:        tail -f /var/log/csm-0.log"
    info "  Health:      curl http://localhost:8080/health"
fi
info "  External:    curl https://tts.getampere.ai/health"
info ""
info "Once health shows {\"status\":\"ok\",\"busy\":false}:"
info "  DUAL_GPU=${DUAL_GPU} | GPUs=${GPU_COUNT}"
info "  Benchmark: python3 /app/benchmark.py"
