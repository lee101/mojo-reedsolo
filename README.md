# mojo-reedsolo

Reed-Solomon error correction for Python with the repeated finite-field work
implemented in [Mojo](https://www.modular.com/mojo). The Python API follows
[`reedsolo` 1.7.0](https://pypi.org/project/reedsolo/) and retains its useful
errors-and-erasures decoding, configurable code parameters, and transparent
chunking for long byte strings.

This is a standalone port, not a binding to the upstream implementation.

## Coverage

The public GF(2^8) API exported by this package is covered by parity tests:

- `RSCodec` encoding, checking, decoding, error correction, erasure correction,
  mixed errata, long-message chunking, `maxerrata`, and custom `nsize`, `fcr`,
  primitive polynomial, and generator parameters.
- The low-level `gf_*` field and polynomial functions.
- The low-level `rs_*` generator, encoding, syndrome, locator, evaluator,
  Forney, correction, and checking functions.
- Representative return containers, failure behavior, argument defaults, and
  the main public signatures are checked against `reedsolo` 1.7.0.

The current boundary is fields larger or smaller than GF(2^8): `c_exp != 8`,
symbols above 255, and codewords above 255 symbols without `RSCodec` chunking
are not implemented. Upstream's prime-polynomial search helper remains
available, but larger-field codec construction raises `NotImplementedError`.
This package also does not expose upstream's module internals or support
strided, multidimensional, floating-point, or out-of-range symbol buffers.

## Install

```bash
pixi install
pixi run build
pixi run test
```

Pixi activates `python/` on `PYTHONPATH`. The build creates
`dist/libmojo-reedsolo.so`.

## Usage

```python
from mojo_reedsolo import RSCodec

codec = RSCodec(10)
encoded = codec.encode(b"hello world")

damaged = bytearray(encoded)
for position in (0, 3, 7, 12, 18):
    damaged[position] ^= 0x55

message, repaired, positions = codec.decode(damaged)
assert message == b"hello world"
assert codec.check(repaired) == [True]
assert len(positions) == 5
```

Known erasures use the same upstream argument:

```python
message, repaired, positions = codec.decode(
    damaged, erase_pos=[0, 3, 7, 12, 18]
)
```

## Benchmarks

Measured with `pixi run bench` on the machine shown below. Times are the best
of three runs and compare against the conda-forge `reedsolo` 1.7.0 package on
identical buffers.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux 6.8.0-136-generic

| operation | mojo-reedsolo | reedsolo 1.7.0 | speedup |
| --- | ---: | ---: | ---: |
| encode 256 KiB, 32 ECC | 23.83 ms | 1298.37 ms | 54.49x |
| check 292.75 KiB codeword | 57.58 ms | 2005.72 ms | 34.83x |
| decode clean 292.75 KiB | 60.01 ms | 2014.57 ms | 33.57x |
| correct 100 blocks, 8 errors | 42.26 ms | 431.22 ms | 10.20x |
| multiply 9x5 polynomials, 10k calls | 76.73 ms | 77.80 ms | 1.01x |
| construct 1k codecs, 32 ECC | 151.33 ms | 374.30 ms | 2.47x |

Run the benchmark on your own machine with:

```bash
pixi run bench
```

## How it works

Encoding uses extended synthetic division over GF(256); checking and the clean
decode path evaluate syndrome polynomials. Those nested loops run in one Mojo
compilation unit. Generator construction crosses the native boundary once and
updates independent coefficient ranges with SIMD gathers, unaligned-safe loads
and stores, and a scalar tail. Chien search uses the same SIMD width. Large
polynomial products use a native SIMD kernel, while tiny products stay on a
validated serial CPU path to avoid FFI setup overhead. Berlekamp-Massey and
Forney correction keep their short per-codeword control flow in Python.

There is intentionally no threaded or GPU path. GF(256) codewords are capped at
255 symbols, so the remaining independent kernels are too small for thread or
device launch overhead. Their work is dominated by table gathers rather than
high-arithmetic-intensity floating-point computation, so a GPU path is not
justified.

Python owns all memory. Inputs and outputs at the native boundary are
contiguous NumPy `uint8` buffers, matching one GF(256) symbol per byte. ctypes
passes their addresses as integer values through a small C ABI; Mojo validates
the addresses and lengths before reconstructing pointers. Bytes, bytearrays,
compatible memoryviews, and field tables use zero-copy NumPy views. Other
integer sequences are range-checked and copied to contiguous `uint8` arrays.
Native output is written directly into its final Python `bytearray`. Local
references keep every buffer alive for the synchronous call. The shared library
neither allocates nor retains Python-owned memory.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The parity suite compares exact encoded bytes, syndromes, corrected messages,
ECC blocks, errata positions, exception behavior, helper functions, and public
signatures against upstream `reedsolo` 1.7.0.

MIT licensed. The port retains attribution for the upstream MIT implementation.
