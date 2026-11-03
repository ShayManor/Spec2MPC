#!/usr/bin/env bash
# One-time setup of the LIBERO client env (idempotent).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
commit=8f1084e3132a39270c3a13ebe37270a43ece2a01
if [ ! -d "$here/LIBERO/.git" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$here/LIBERO"
fi
git -C "$here/LIBERO" fetch --depth 1 origin "$commit" 2>/dev/null || true
git -C "$here/LIBERO" checkout -q "$commit"
# LIBERO prompts on first import unless a config exists; point it at the clone
root="$here/LIBERO/libero/libero"
mkdir -p "$here/config"
cat > "$here/config/config.yaml" <<YAML
benchmark_root: $root
bddl_files: $root/bddl_files
init_states: $root/init_files
datasets: $root/../datasets
assets: $root/assets
YAML
uv="${UV:-$HOME/.local/bin/uv}"
"$uv" sync --project "$here"
site="$("$here/.venv/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$here/LIBERO" > "$site/libero_src.pth"
