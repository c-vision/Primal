#!/bin/sh
# make_gifs.sh FRAMES_DIR OUT_DIR GIFJOIN
#
# Turns the frame lists written by make_animations (FRAMES_DIR/<name>.frames, one "CS file.svg"
# line per frame) into OUT_DIR/<name>.gif.  Needs rsvg-convert (SVG -> PNG), sips (PNG -> GIF, a
# macOS tool) and the gifjoin program built from gifjoin.c.  No Python is involved.
set -e
frames=$1; out=$2; join=$3
[ -d "$frames" ] && [ -d "$out" ] && [ -x "$join" ] || { echo "usage: make_gifs.sh FRAMES_DIR OUT_DIR GIFJOIN" >&2; exit 2; }
for list in "$frames"/*.frames; do
    name=$(basename "$list" .frames)
    args=""
    while read -r cs svg; do
        base=${svg%.svg}
        rsvg-convert -o "$frames/$base.png" "$frames/$svg"
        sips -s format gif "$frames/$base.png" --out "$frames/$base.gif" >/dev/null
        args="$args $cs:$frames/$base.gif"
    done < "$list"
    # shellcheck disable=SC2086
    "$join" "$out/$name.gif" $args
    echo "wrote $out/$name.gif ($(wc -l < "$list") frames)"
done
