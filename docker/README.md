# Background-run options for v4

You don't want to keep SSH open for 25 hours. Pick one of the three
options below in order of "robustness vs setup effort":

| Option | Setup time | Survives SSH disconnect | Survives host reboot | Reproducible env |
|---|---|---|---|---|
| 1. `tmux` | 10 s | ✓ | ✗ | ✗ (uses host pip env) |
| 2. `nohup` + `&` | 5 s | ✓ | ✗ | ✗ (uses host pip env) |
| 3. **Docker** (this dir) | ~15 min build | ✓ | ✓ (with `--restart`) | ✓ |

---

## Option 1: `tmux` (recommended for "just get it running tonight")

`tmux` is a "screen multiplexer" — it runs a shell session on the host that
keeps going after you disconnect. Almost certainly already installed.

```bash
# On the T4 server
tmux new -s v4                                   # start a new named session

# Inside tmux:
cd ~/projects/python/OpenADMET-Reproduction
python -m src.v4_hybridadmet.run 2>&1 | tee v4_run.log

# Press Ctrl+B then D to "detach" — leaves the run going
# Now you can `exit` your SSH safely.

# Later, to reconnect:
ssh ...
tmux attach -t v4                                # back in the same session
```

Pros: instant, no Docker, the run keeps going. Cons: if the VM reboots,
you lose the run; no env isolation.

---

## Option 2: `nohup` (one-liner)

```bash
cd ~/projects/python/OpenADMET-Reproduction
nohup python -m src.v4_hybridadmet.run > v4_run.log 2>&1 &
echo $! > v4_run.pid                              # save the PID so you can kill it later

# safely `exit`
# come back later:
tail -f v4_run.log                                # tail the log
ps -p $(cat v4_run.pid)                           # check it's still alive
kill $(cat v4_run.pid)                            # stop it
```

Pros: shortest possible. Cons: no env isolation; the process dies if the
VM reboots.

---

## Option 3: Docker (best for reproducibility + host-reboot survival)

```bash
# One-time build (~15 min, ~8 GB image)
cd ~/projects/python/OpenADMET-Reproduction
./docker/run_v4.sh build

# Start the training in the background
./docker/run_v4.sh up
# SSH can disconnect now. The container restarts on host reboot.

# Later:
./docker/run_v4.sh logs                # tail training logs
./docker/run_v4.sh status              # is it still running?
./docker/run_v4.sh stop                # stop and remove

# Need a shell inside the image (e.g. to debug)?
./docker/run_v4.sh shell
```

Equivalent with `docker compose`:

```bash
cd docker
docker compose up -d v4
docker compose logs -f v4
docker compose down
```

### What's mounted where

- `output/` (on host) → `/workspace/output` (in container) — checkpoints +
  submission CSVs persist.
- The project source on host → `/workspace` — code edits take effect
  next container start without rebuilding.
- `../ExpansionRx-Challenge-Eval` (host) → `/eval` (read-only) — for the
  official `python -m eval` scorer. Override with
  `EVAL_REPO_HOST=/abs/path/to/eval ./docker/run_v4.sh up`.
- `openadmet-cache` named volume → `/root/.cache` — Uni-Mol2 + chemprop
  + huggingface checkpoint cache, survives across containers.
- `openadmet-chemprop` named volume → `/root/.chemprop` — the
  CheMeleon foundation checkpoint, ~350 MB, downloaded once.

### Tuning runtime params at start time

The container reads these environment vars (see `Dockerfile`/compose for
defaults):

```bash
V4_EPOCHS=30 V4_ENS=3 UNIMOL_FINETUNE=1 \
    ./docker/run_v4.sh up
```

Lower epochs / smaller ensemble = faster but worse. Quick "smoke test"
config that finishes in ~3 h:

```bash
V4_EPOCHS=20 V4_BATCH=64 V4_ENS=2 UNIMOL_FINETUNE=0 \
    ./docker/run_v4.sh up
```

### Host requirements

- Linux + an NVIDIA GPU
- Docker Engine 20.10+
- **`nvidia-container-toolkit`** installed (this is what gives `--gpus`
  support). On Ubuntu:
  ```bash
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
      | sudo sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt update && sudo apt install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

Verify with:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```
Should print your T4 with no errors.

### Troubleshooting

- **"could not select device driver" / "nvidia"** — the host doesn't have
  `nvidia-container-toolkit` set up. Fall back to Option 1 (`tmux`) until
  you can install it.
- **Image build fails on `torch-scatter`** — the version of torch-scatter
  in PyPI doesn't match your CUDA. Edit the `Dockerfile` to pin
  `torch==X.Y.Z` and update the `-f https://data.pyg.org/whl/...` URL to
  the matching pyg wheel index.
- **Container OOM during Uni-Mol2 fine-tune** — T4 has 16 GB; drop
  `V4_BATCH` to 16 or set `UNIMOL_FINETUNE=0` (uses Uni-Mol2 as a frozen
  feature extractor, much cheaper).
- **`/workspace/output` is owned by root after a container run** — that's
  Docker's default. Fix once with:
  ```bash
  sudo chown -R $(id -u):$(id -g) output/
  ```
  Or add `--user $(id -u):$(id -g)` to the `docker run` line in
  `run_v4.sh`.
