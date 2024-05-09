# Monolith Plonky2 with Acceleration

This repo is a fork of [Monolith](https://github.com/HorizenLabs/monolith) on which we add the AVX2 implementation for the [Monolith hash function](https://eprint.iacr.org/2023/1025.pdf).

## Run

This repo needs Rust nightly release.

```
rustup override set nightly
```

## Benchmark

Without AVX:

```
cargo bench --bench merkle
```

With AVX support:

```
RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench merkle
```

## Performance Results for Merkle Tree Building

On a system with AMD Ryzen 9 7950X @ 5 GHz and DDR5 RAM.


Tree Size (leaves) | Poseidon (no AVX) | Monolith (no AVX) | Monolith (AVX) |
--- | --- | --- | ---
2^13 | 16.2 ms | 8.7 ms | 8.1 ms
2^14 | 33.3 ms | 18.7 ms | 17.3 ms
2^15 | 67.2 ms | 37.9 ms | 35.1 ms