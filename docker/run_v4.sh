#!/usr/bin/env bash
# One-shot launcher for the v4 pipeline that works WITHOUT docker compose
# (useful on machines where you just have plain `docker` + nvidia-container-toolkit).
#
#   ./docker/run_v4.sh                  # build (if needed) + run detached
#   ./docker/run_v4.sh logs             # tail the running container's logs
#   ./docker/run_v4.sh stop             # stop + remove the running container
#   ./docker/run_v4.sh shell            # interactive bash inside a fresh container

set -euo pipefail

# --- find paths ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="${IMAGE:-openadmet-v4:latest}"
NAME="${NAME:-openadmet-v4}"

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
        # Build only if image doesn't exist
        docker image inspect "$IMAGE" > /dev/null 2>&1 || {
            echo "Image $IMAGE not found — building"
            cd "$PROJ_DIR"
            docker build -f docker/Dockerfile -t "$IMAGE" .
        }
        # Stop existing container with same name
        docker rm -f "$NAME" 2>/dev/null || true
        echo "Starting $NAME (detached). Logs: ./docker/run_v4.sh logs"
        docker run -d --name "$NAME" \
            --gpus all \
            --shm-size=8g \
            --restart unless-stopped \
            -v "$PROJ_DIR:/workspace" \
            -v "$PROJ_DIR/output:/workspace/output" \
            -v "$EVAL_REPO_HOST:/eval:ro" \
            -v openadmet-cache:/root/.cache \
            -v openadmet-chemprop:/root/.chemprop \
            -e V4_EPOCHS="${V4_EPOCHS:-60}" \
            -e V4_BATCH="${V4_BATCH:-32}" \
            -e V4_ENS="${V4_ENS:-5}" \
            -e UNIMOL_FINETUNE="${UNIMOL_FINETUNE:-1}" \
            -e UNIMOL_MODEL="${UNIMOL_MODEL:-84M}" \
            -e EVAL_REPO="/eval" \
            -e PYTHONPATH="/workspace" \
            "$IMAGE" v4
        echo
        echo "✓ Container '$NAME' started. SSH can now disconnect."
        echo "  Tail logs:    ./docker/run_v4.sh logs"
        echo "  Stop:         ./docker/run_v4.sh stop"
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
        echo "  up      start the v4 training container in background (DEFAULT)"
        echo "  logs    tail running container's logs"
        echo "  stop    stop and remove the container"
        echo "  shell   interactive bash inside a fresh container"
        echo "  status  show container status"
        exit 1
        ;;
esac
