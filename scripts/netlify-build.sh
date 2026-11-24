#!/usr/bin/env bash
# Netlify build: compile the Rust core to WebAssembly into web/pkg.
#
# Netlify's build image ships rustup; rust-toolchain.toml pins the Rust
# version and adds the wasm32 target automatically. wasm-pack is downloaded
# as a pinned prebuilt binary (seconds, instead of minutes with cargo install).
set -euo pipefail

WASM_PACK_VERSION="0.15.1"

# Make sure the toolchain from rust-toolchain.toml (with its wasm target) is in place.
rustup show active-toolchain || rustup toolchain install
rustc --version

if ! command -v wasm-pack >/dev/null 2>&1; then
  dir="$(mktemp -d)"
  curl -sSfL "https://github.com/wasm-bindgen/wasm-pack/releases/download/v${WASM_PACK_VERSION}/wasm-pack-v${WASM_PACK_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xz -C "$dir"
  export PATH="$dir/wasm-pack-v${WASM_PACK_VERSION}-x86_64-unknown-linux-musl:$PATH"
fi
wasm-pack --version

wasm-pack build wasm --release --target web --out-dir ../web/pkg
ls -l web/pkg
