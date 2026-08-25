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
uv venv "$build_root/export-venv" --python 3.12 --clear
export_python="$build_root/export-venv/bin/python"
uv pip install --python "$export_python" torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$export_python" onnx
"$export_python" "$repo_root/scripts/fetch_release_models.py" --output "$repo_root/models"
runtime_models="$build_root/models"
"$export_python" "$repo_root/scripts/export_onnx_runtime.py" \
  "$repo_root/models/v8-hf3k-face-gan.pt" \
  "$repo_root/models/v8-flex8k-ota-rxfix.pt" \
  --output "$runtime_models"

hamlib_dir="$build_root/hamlib"
mkdir -p "$hamlib_dir"
cp "$(command -v rigctl)" "$hamlib_dir/rigctl"
hamlib_library="$(ldconfig -p | awk '/libhamlib\.so\.4/ && !found {found=$NF} END {print found}')"
if [[ -z "$hamlib_library" ]]; then
  echo "libhamlib.so.4 is not installed" >&2
  exit 1
fi
cp -L "$hamlib_library" "$hamlib_dir/libhamlib.so.4"
cp /usr/share/common-licenses/LGPL-2.1 "$hamlib_dir/COPYING.LIB.txt"
cp /usr/share/common-licenses/GPL-2 "$hamlib_dir/COPYING.txt"
cp /usr/share/doc/libhamlib4t64/copyright "$hamlib_dir/HAMLIB-COPYRIGHT.txt"

uv venv "$build_root/runtime-venv" --python 3.12 --clear
python_bin="$build_root/runtime-venv/bin/python"
uv pip install --python "$python_bin" "$repo_root[gui]" pyinstaller

rm -rf "$dist_root"
mkdir -p "$dist_root"
work_path="$build_root/pyinstaller"
spec_path="$build_root/spec"
common=(
  --noconfirm --clean --onedir
  --workpath "$work_path"
  --specpath "$spec_path"
  --distpath "$dist_root"
  --exclude-module torch
  --exclude-module torchvision
  --exclude-module aetv.models
  --exclude-module aetv.channel
  --exclude-module aetv.data
  --exclude-module aetv.video_backbone
  --add-data "$repo_root/aetv/assets:aetv/assets"
  --add-data "$hamlib_dir:aetv/bin"
)
for model in "$runtime_models"/*; do
  common+=(--add-data "$model:models")
done

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
  XDG_CONFIG_HOME="$build_root/smoke-config" \
    QT_QPA_PLATFORM=offscreen AETV_OFFLINE=1 ./AETV --smoke-test
)

packaged_models="$app_dir/_internal/models"
case "$packaged_models" in
  "$app_dir"/_internal/models) rm -rf "$packaged_models" ;;
  *) echo "Refusing to remove models outside packaged app" >&2; exit 1 ;;
esac

archive="$dist_root/AETV-linux-x64-cpu.tar.gz"
tar -czf "$archive" -C "$dist_root" AETV
echo "Portable AETV Linux build: $archive"
