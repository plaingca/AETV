#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="$repo_root/.build/linux-cpu"
dist_root="$repo_root/dist/linux-cpu"

case "$build_root" in
  "$repo_root"/.build/*) ;;
  *) echo "Refusing to build outside the repository" >&2; exit 1 ;;
esac
case "$dist_root" in
  "$repo_root"/dist/*) ;;
  *) echo "Refusing to package outside the repository" >&2; exit 1 ;;
esac

mkdir -p "$build_root"
uv venv "$build_root/venv" --python 3.12 --clear
python_bin="$build_root/venv/bin/python"
uv pip install --python "$python_bin" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$python_bin" "$repo_root[gui]" pyinstaller
"$python_bin" "$repo_root/scripts/fetch_release_models.py" --output "$repo_root/models"

rm -rf "$dist_root"
mkdir -p "$dist_root"
work_path="$build_root/pyinstaller"
spec_path="$build_root/spec"
common=(
  --noconfirm --clean --onedir
  --workpath "$work_path"
  --specpath "$spec_path"
  --distpath "$dist_root"
  --add-data "$repo_root/models/v8-hf3k-face-gan.pt:models"
  --add-data "$repo_root/models/v8-flex8k-ota-rxfix.pt:models"
  --add-data "$repo_root/aetv/assets:aetv/assets"
)

"$python_bin" -m PyInstaller "${common[@]}" --windowed --name AETV \
  --icon "$repo_root/aetv/assets/aetv-logo.png" \
  "$repo_root/aetv/gui/app.py"
"$python_bin" -m PyInstaller "${common[@]}" --console --name AETV-Benchmark \
  "$repo_root/scripts/benchmark_inference.py"

app_dir="$dist_root/AETV"
cp "$dist_root/AETV-Benchmark/AETV-Benchmark" "$app_dir/AETV-Benchmark"
cp "$repo_root/README.md" "$repo_root/LICENSE" "$repo_root/NOTICE" "$app_dir/"

(
  cd "$app_dir"
  AETV_OFFLINE=1 ./AETV-Benchmark \
    --mode V8 --device cpu --warmup 0 --repeats 1 --json build-smoke.json
)

archive="$dist_root/AETV-linux-x64-cpu.tar.gz"
tar -czf "$archive" -C "$dist_root" AETV
echo "Portable AETV Linux build: $archive"
