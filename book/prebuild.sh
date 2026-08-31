#!/usr/bin/env bash
# Generate figures and other build-time content that used to be produced by the
# `mdbook-cmdrun` preprocessor (dropped: no mdBook 0.5 release, upstream issue #22).
# Run before `mdbook build`. Resolves paths relative to this script, so it can be
# invoked from anywhere.
set -euo pipefail

book="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$book")"

# --- mdbook-langtabs assets (langtabs.css / langtabs.js) --------------------
# git-ignored; regenerated here so `mdbook build` finds them. Leaves book.toml
# untouched when its [preprocessor.langtabs] / additional-* keys already exist.
mdbook-langtabs install "$root"

# --- figures (matplotlib -> SVG) ---------------------------------------------
# A figure script writes SVG(s) that a chapter then inlines via {{#include}}.
# fixup(): swap matplotlib's black inks for mdBook theme variables, and drop the
# `<?xml?>` / `<!DOCTYPE>` prologue that mdBook 0.5's HTML parser rejects.
fixup() {
    sed -i 's/#000000/var(--fg)/g;s/#000002/var(--icons)/g;s/#000005/var(--sidebar-active)/g;/^<svg/,$!d' "$@"
}

( cd "$book/math"
  python3 special/figures.py
  python3 integrate/figures.py )
fixup "$book"/math/special/*.svg "$book"/math/integrate/*/*.svg

# Reproduction figures: one script per figure, run from its own directory so a
# bare `plt.savefig("figure_1.svg", ...)` lands next to it (as in math/*/figures.py).
while IFS= read -r -d '' script; do
    ( cd "$(dirname "$script")" && python3 "$(basename "$script")" )
done < <(find "$book/reproductions" -name '*.py' -print0)
while IFS= read -r -d '' svg; do
    fixup "$svg"
done < <(find "$book/reproductions" -name '*.svg' -print0)

# --- contributors grid (GitHub API) ----------------------------------------
# Network-dependent; on failure emit a placeholder so the build still succeeds.
cd "$book"
if ! python3 contributors.py > contributors.generated.md; then
    echo "<!-- contributors.py failed; grid omitted -->" > contributors.generated.md
fi
