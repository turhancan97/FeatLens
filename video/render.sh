#!/usr/bin/env bash
# Render every scene at 1080p60 and join them into video/renders/featlens_explainer.mp4.
#
#   MANIM_ENV=/path/to/env  bash video/render.sh  [-ql]     # -ql for a fast 480p preview
#   SKIP_RENDER=1 ...                                      # only re-join existing scene clips
#
# MANIM_ENV must have manim, kokoro, soundfile and ffmpeg. Run prepare_data.py and
# prepare_scenes.py first (in an env with featlens + timm) to build video/assets/.
set -euo pipefail

ENV_BIN="${MANIM_ENV:?set MANIM_ENV to the env with manim + kokoro}/bin"
QUALITY="${1:--qh}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MEDIA="${MEDIA_DIR:-$HERE/renders/media}"
export PATH="$ENV_BIN:$PATH"

render() { [ -n "${SKIP_RENDER:-}" ] || manim "$QUALITY" --media_dir "$MEDIA" "$HERE/$1" "$2"; }
render scenes.py ColdOpen
render scenes.py TokensScene
render scene3_pca.py PCAScene
render scenes.py MethodsScene
render scenes.py BeyondScene
render scenes.py CodeScene

case "$QUALITY" in -ql) RES=480p15 ;; -qm) RES=720p30 ;; *) RES=1080p60 ;; esac
INPUTS=()
FILTER=""
SRT_SPECS=()
i=0
for clip in scenes/ColdOpen scenes/TokensScene scene3_pca/PCAScene scenes/MethodsScene \
            scenes/BeyondScene scenes/CodeScene; do
  file="$MEDIA/videos/${clip%%/*}/$RES/${clip##*/}.mp4"
  INPUTS+=(-i "$file")
  # Each scene's narration ends before its picture does; pad it with silence to exactly the
  # scene's video length so the next scene's audio starts in sync with its picture.
  vdur=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$file")
  FILTER+="[$i:a]apad,atrim=0:$vdur,asetpts=PTS-STARTPTS[a$i];"
  SRT_SPECS+=("${clip##*/}:$vdur")
  i=$((i + 1))
done
for j in $(seq 0 $((i - 1))); do FILTER+="[$j:v][a$j]"; done

# Subtitles: phrase timings logged during rendering -> one .srt for the joined video.
SRT="$HERE/renders/featlens_explainer.srt"
python3 "$HERE/make_srt.py" "$SRT" "${SRT_SPECS[@]}"

# Burn the .srt in with libass (white DejaVu Sans on a translucent black box). Also write a
# clean copy without burned-in subtitles, for platforms that take the .srt separately.
STYLE="FontName=DejaVu Sans,FontSize=15,PrimaryColour=&H00F7F5F4,BorderStyle=3,Outline=1.6,OutlineColour=&H55000000,Shadow=0,MarginV=12"
FILTER+="concat=n=$i:v=1:a=1[v][a];[v]split[v1][v2];[a]asplit[a1][a2];[v1]subtitles=filename='$SRT':force_style='$STYLE'[vs]"
ENC=(-c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a aac -b:a 192k -movflags +faststart)
ffmpeg -v error -y "${INPUTS[@]}" -filter_complex "$FILTER" \
  -map "[vs]" -map "[a1]" "${ENC[@]}" "$HERE/renders/featlens_explainer.mp4" \
  -map "[v2]" -map "[a2]" "${ENC[@]}" "$HERE/renders/featlens_explainer_nosubs.mp4"
echo "wrote $HERE/renders/featlens_explainer.mp4 (subtitled) + featlens_explainer_nosubs.mp4"
