#!/bin/sh

set -e
cd "$(dirname "$0")/.."
output=src/domain/gpu/kernels.ptx
cargo +nightly build --release --example kernels --target nvptx64-nvidia-cuda \
    --no-default-features --features kernels
built=target/nvptx64-nvidia-cuda/release/examples/kernels.ptx
echo "kernels:"
grep -o "^\.visible \.entry [A-Za-z0-9_]*" "$built" | sed 's/^\.visible \.entry /  /'
if [ "$1" = "--check" ]; then
    if cmp -s "$built" "$output"; then
        echo "kernels.ptx is the one that the sources build"
    else
        echo "kernels.ptx is not the one that the sources build: run kernels/build.sh"
        exit 1
    fi
else
    mkdir -p "$(dirname "$output")"
    cp "$built" "$output"
    echo "wrote $output"
fi
