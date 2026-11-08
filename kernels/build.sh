#!/bin/sh
# Builds the PTX that the `gpu` feature of the library embeds, from kernels/lib.rs, with the
# nightly toolchain, into src/domain/gpu/kernels.ptx, which git ignores and a release publishes.
# It needs the target and the linker of the device:
#   rustup target add nvptx64-nvidia-cuda --toolchain nightly
#   rustup component add llvm-bitcode-linker --toolchain nightly
#
#   kernels/build.sh          builds the PTX and puts it where the library takes it from
#   kernels/build.sh --check  builds it and says whether it is the one that is there, for CI
set -e
cd "$(dirname "$0")/.."
output=src/domain/gpu/kernels.ptx
listing=src/domain/gpu/kernels.d
cargo +nightly build --release --example kernels --target nvptx64-nvidia-cuda \
    --no-default-features --features kernels
built=target/nvptx64-nvidia-cuda/release/examples/kernels.ptx
sources=target/nvptx64-nvidia-cuda/release/examples/kernels.d
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
    # The files that it was built from, with which the library knows when it is out of date.
    cp "$sources" "$listing"
    echo "wrote $output"
fi
