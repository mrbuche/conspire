#!/bin/sh
# Builds the PTX of the kernels of the fit benchmark, from kernels/fit.rs, with the nightly
# toolchain. The benchmark, tests/fit_gpu_bench.rs, reads it from where it is built, or from the
# file in FIT_KERNELS. It needs the target and the linker of the device:
#   rustup target add nvptx64-nvidia-cuda --toolchain nightly
#   rustup component add llvm-bitcode-linker --toolchain nightly
set -e
cd "$(dirname "$0")/.."
cargo +nightly build --release --example fit_kernels --target nvptx64-nvidia-cuda \
    --no-default-features --features kernels
built=target/nvptx64-nvidia-cuda/release/examples/fit_kernels.ptx
echo "kernels:"
grep -o "^\.visible \.entry [A-Za-z0-9_]*" "$built" | sed 's/^\.visible \.entry /  /'
echo "built $built"
