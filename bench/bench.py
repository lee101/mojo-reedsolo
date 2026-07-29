from __future__ import annotations

import os
import platform
import random
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojo_reedsolo as mojo
import reedsolo as reference


def best_time(fn, repeats=3):
    best = float("inf")
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def format_ms(seconds):
    return f"{seconds * 1000:.2f} ms"


def row(name, ours, theirs):
    ratio = theirs / ours
    print(
        f"| {name} | {format_ms(ours)} | {format_ms(theirs)} | {ratio:.2f}x |"
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    nsym = 32
    payload = random.Random(2026).randbytes(256 * 1024)
    ours = mojo.RSCodec(nsym)
    theirs = reference.RSCodec(nsym)
    ours_encoded = ours.encode(payload)
    ref_encoded = theirs.encode(payload)
    assert ours_encoded == ref_encoded

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| operation | mojo-reedsolo | reedsolo 1.7.0 | speedup |")
    print("| --- | ---: | ---: | ---: |")

    mt, _ = best_time(lambda: ours.encode(payload))
    rt, _ = best_time(lambda: theirs.encode(payload))
    row("encode 256 KiB, 32 ECC", mt, rt)

    mt, _ = best_time(lambda: ours.check(ours_encoded))
    rt, _ = best_time(lambda: theirs.check(ref_encoded))
    row("check 292.75 KiB codeword", mt, rt)

    mt, _ = best_time(lambda: ours.decode(ours_encoded))
    rt, _ = best_time(lambda: theirs.decode(ref_encoded))
    row("decode clean 292.75 KiB", mt, rt)

    block = bytearray(ours_encoded[:255])
    for pos in [1, 17, 44, 83, 121, 166, 201, 240]:
        block[pos] ^= 0xA5

    def mojo_correct():
        for _ in range(100):
            ours.decode(block)

    def reference_correct():
        for _ in range(100):
            theirs.decode(block)

    mt, _ = best_time(mojo_correct)
    rt, _ = best_time(reference_correct)
    row("correct 100 blocks, 8 errors", mt, rt)


if __name__ == "__main__":
    main()
