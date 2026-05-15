#!/usr/bin/env bash
# Docker entrypoint for openadmet-v4 image.
#   docker run ... openadmet-v4:latest              # default: run v4 pipeline
#   docker run ... openadmet-v4:latest v4           # explicit v4
#   docker run ... openadmet-v4:latest v3           # run v3 pipeline instead
#   docker run ... openadmet-v4:latest bash         # interactive shell
#   docker run ... openadmet-v4:latest python ...   # arbitrary python cmd
set -euo pipefail

cd /workspace

# Optional: pull latest project code if a git remote is reachable. Lets you
# update without rebuilding the image.
if [[ "${V4_GIT_PULL:-0}" == "1" ]] && [[ -d /workspace/.git ]]; then
    echo "[entrypoint] git pull"
    git pull --ff-only || echo "[entrypoint] git pull failed; using image code"
fi

CMD="${1:-v4}"
shift || true

case "$CMD" in
    v4)
        echo "[entrypoint] running: python -m src.v4_hybridadmet.run $@"
        exec python -m src.v4_hybridadmet.run "$@"
        ;;
    v3)
        echo "[entrypoint] running: python -m src.v3.run $@"
        exec python -m src.v3.run "$@"
        ;;
    bash|sh)
        exec bash
        ;;
    python|python3)
        exec python "$@"
        ;;
    *)
        # Fallback: execute whatever the user passed
        exec "$CMD" "$@"
        ;;
esac
