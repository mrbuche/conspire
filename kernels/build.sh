#!/bin/sh

set -e
cd "$(dirname "$0")/.."
check=
if [ "$1" = "--check" ]; then
    check=1
    shift
fi
applications=${*:-$(ls kernels/*/lib.rs | sed 's|kernels/\(.*\)/lib.rs|\1|')}
status=0
for application in $applications; do
    output=src/gpu/kernels/$application.ptx
    cargo +nightly build --release --example kernels_"$application" --target nvptx64-nvidia-cuda \
        --no-default-features --features kernels
    built=target/nvptx64-nvidia-cuda/release/examples/kernels_$application.ptx
    echo "$application:"
    grep -o "^\.visible \.entry [A-Za-z0-9_]*" "$built" | sed 's/^\.visible \.entry /  /'
    if [ -n "$check" ]; then
        if cmp -s "$built" "$output"; then
            echo "$output is the one that the sources build"
        else
            echo "$output is not the one that the sources build: run kernels/build.sh"
            status=1
        fi
    else
        mkdir -p "$(dirname "$output")"
        cp "$built" "$output"
        echo "wrote $output"
    fi
done
exit $status
