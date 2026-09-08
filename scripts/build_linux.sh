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
runtime_models="$build_root/models"
"$python_bin" "$repo_root/scripts/fetch_release_runtime.py" --output "$runtime_models"

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
  --exclude-module imageio_ffmpeg
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
ffmpeg_source="$("$python_bin" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
if [[ ! -x "$ffmpeg_source" ]]; then
  echo "imageio-ffmpeg did not provide a Linux executable: $ffmpeg_source" >&2
  exit 1
fi
cp "$ffmpeg_source" "$app_dir/ffmpeg"
chmod 755 "$app_dir/ffmpeg"
cp "$dist_root/AETV-Benchmark/AETV-Benchmark" "$app_dir/AETV-Benchmark"
cp "$repo_root/README.md" "$repo_root/LICENSE" "$repo_root/NOTICE" \
  "$repo_root/FFMPEG-NOTICE.txt" "$app_dir/"

(
  cd "$app_dir"
  ./AETV-Benchmark --video-save-smoke "$build_root/saved-video-smoke.mp4"
  test -s "$build_root/saved-video-smoke.mp4"
  XDG_CACHE_HOME="$build_root/smoke-cache" \
    AETV_OFFLINE=1 ./AETV-Benchmark \
    --mode V8 --device cpu --warmup 0 --repeats 1 --json build-smoke.json
  XDG_CACHE_HOME="$build_root/smoke-cache" \
    XDG_CONFIG_HOME="$build_root/smoke-config" \
    QT_QPA_PLATFORM=offscreen AETV_OFFLINE=1 ./AETV --smoke-test \
      --video-smoke-output "$build_root/saved-video-smoke.mp4"
  test -s "$build_root/saved-video-smoke.mp4"
)

packaged_models="$app_dir/_internal/models"
case "$packaged_models" in
  "$app_dir"/_internal/models) rm -rf "$packaged_models" ;;
  *) echo "Refusing to remove models outside packaged app" >&2; exit 1 ;;
esac

appimage_tool="${APPIMAGETOOL:-appimagetool}"
if ! command -v "$appimage_tool" >/dev/null 2>&1 && [[ ! -x "$appimage_tool" ]]; then
  echo "appimagetool was not found; set APPIMAGETOOL to its executable path" >&2
  exit 1
fi

appimage_dir="$build_root/AETV.AppDir"
case "$appimage_dir" in
  "$repo_root"/.build/*) rm -rf "$appimage_dir" ;;
  *) echo "Refusing to prepare an AppDir outside the repository" >&2; exit 1 ;;
esac
mkdir -p "$appimage_dir/usr/lib"
cp -a "$app_dir" "$appimage_dir/usr/lib/AETV"
cat > "$appimage_dir/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/lib/AETV/AETV" "$@"
EOF
chmod 755 "$appimage_dir/AppRun"
cat > "$appimage_dir/aetv.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=AETV
Comment=Live learned video over amateur-radio channels
Exec=AETV
Icon=aetv
Terminal=false
Categories=AudioVideo;HamRadio;
EOF
cp "$repo_root/aetv/assets/aetv-logo.png" "$appimage_dir/aetv.png"

appimage="$dist_root/AETV-linux-x64-cpu.AppImage"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$appimage_tool" "$appimage_dir" "$appimage"
chmod 755 "$appimage"

archive="$dist_root/AETV-linux-x64-cpu.tar.gz"
tar -czf "$archive" -C "$dist_root" AETV
echo "Portable AETV Linux build: $archive"
echo "AETV Linux AppImage: $appimage"
