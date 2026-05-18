#!/usr/bin/env bash
# One-shot launcher for the v5 (Uni-Mol2-only) pipeline.
#
# Reuses the openadmet-v4 image — v5 doesn't need anything v4 doesn't have,
# and skipping the second build saves ~20 minutes + a second 8 GB image.
# Just runs the container with command "v5" so entrypoint.sh routes to
# python -m src.v5_unimol.run.
#
#   ./docker/run_v5.sh                  # build image (if needed) + run detached
#   ./docker/run_v5.sh logs             # tail the running container's logs
#   ./docker/run_v5.sh stop             # stop + remove the running container
#   ./docker/run_v5.sh shell            # interactive bash inside a fresh container

set -euo pipefail

# --- find paths ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="${IMAGE:-openadmet-v4:latest}"        # share with v4
NAME="${NAME:-openadmet-v5}"                 # but distinct container name

# Where the official-eval repo lives on this host (override with $EVAL_REPO_HOST)
EVAL_REPO_HOST="${EVAL_REPO_HOST:-$(dirname "$PROJ_DIR")/ExpansionRx-Challenge-Eval}"

CMD="${1:-up}"
shift || true

case "$CMD" in
    build)
        echo "Building $IMAGE …"
        cd "$PROJ_DIR"
        docker build -f docker/Dockerfile -t "$IMAGE" .
        ;;
    up|run|start)
        docker image inspect "$IMAGE" > /dev/null 2>&1 || {
            echo "Image $IMAGE not found — building"
            cd "$PROJ_DIR"
            docker build -f docker/Dockerfile -t "$IMAGE" .
        }
        # Stop existing container with same name
        docker rm -f "$NAME" 2>/dev/null || true
        echo "Starting $NAME (detached, running v5). Logs: ./docker/run_v5.sh logs"
        docker run -d --name "$NAME" \
            --gpus all \
            --shm-size=8g \
            --restart on-failure:5 \
            \
            `# Restart only on CRASH (non-zero exit), max 5 attempts.` \
            `# This avoids the v4-era bug where unless-stopped re-ran the` \
            `# whole 25-h training from scratch after each successful` \
            `# completion.` \
            -v "$PROJ_DIR:/workspace" \
            -v "$PROJ_DIR/output:/workspace/output" \
            -v "$EVAL_REPO_HOST:/eval:ro" \
            -v openadmet-cache:/root/.cache \
            -v openadmet-chemprop:/root/.chemprop \
            -e V5_EPOCHS="${V5_EPOCHS:-60}" \
            -e V5_BATCH="${V5_BATCH:-32}" \
            -e V5_ENS="${V5_ENS:-5}" \
            -e V5_LR="${V5_LR:-1e-4}" \
            -e UNIMOL_FINETUNE="${UNIMOL_FINETUNE:-1}" \
            -e UNIMOL_MODEL="${UNIMOL_MODEL:-84M}" \
            -e EVAL_REPO="/eval" \
            -e PYTHONPATH="/workspace" \
            "$IMAGE" v5
        echo
        echo "✓ Container '$NAME' started. SSH can now disconnect."
        echo "  Tail logs:    ./docker/run_v5.sh logs"
        echo "  Stop:         ./docker/run_v5.sh stop"
        ;;
    logs)
        docker logs -f "$NAME"
        ;;
    stop|down)
        docker rm -f "$NAME" || true
        echo "Stopped $NAME"
        ;;
    shell|bash)
        cd "$PROJ_DIR"
        docker run -it --rm \
            --gpus all --shm-size=8g \
            -v "$PROJ_DIR:/workspace" \
            -v "$EVAL_REPO_HOST:/eval:ro" \
            -v openadmet-cache:/root/.cache \
            -v openadmet-chemprop:/root/.chemprop \
            -e PYTHONPATH="/workspace" \
            "$IMAGE" bash
        ;;
    status|ps)
        docker ps -a --filter "name=$NAME"
        ;;
    *)
        echo "Usage: $0 {build|up|logs|stop|shell|status}"
        echo
        echo "  build   build the image"
        echo "  up      start the v5 training container in background (DEFAULT)"
        echo "  logs    tail running container's logs"
        echo "  stop    stop and remove the container"
        echo "  shell   interactive bash inside a fresh container"
        echo "  status  show container status"
        exit 1
        ;;
esac
