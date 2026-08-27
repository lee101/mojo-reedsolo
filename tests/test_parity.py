import inspect
import random

import numpy as np
import pytest
import reedsolo as upstream

import mojo_reedsolo as mojo


def reset():
    upstream.init_tables()
    mojo.init_tables()


@pytest.mark.parametrize("nsym", [2, 4, 10, 32, 64])
def test_encode_matches_upstream(nsym):
    reset()
    data = bytes((i * 37 + nsym) % 256 for i in range(255 - nsym))
    assert mojo.rs_encode_msg(data, nsym) == upstream.rs_encode_msg(data, nsym)


@pytest.mark.parametrize("nsym", [4, 10, 32])
def test_syndromes_match_upstream(nsym):
    reset()
    data = bytes((i * 19 + 7) % 256 for i in range(255 - nsym))
    encoded = upstream.rs_encode_msg(data, nsym)
    encoded[3] ^= 91
    encoded[-2] ^= 33
    assert mojo.rs_calc_syndromes(encoded, nsym) == upstream.rs_calc_syndromes(
        encoded, nsym
    )


def test_published_hello_world_vector():
    codec = mojo.RSCodec(10)
    assert codec.encode(b"hello world") == bytearray(
        b"hello world\xed%T\xc4\xfd\xfd\x89\xf3\xa8\xaa"
    )


def test_decode_five_errors_matches_upstream():
    ours, reference = mojo.RSCodec(10), upstream.RSCodec(10)
    encoded = ours.encode(b"hello world")
    corrupted = bytearray(encoded)
    for pos in [0, 1, 2, 9, 15]:
        corrupted[pos] ^= 0x5A
    assert ours.decode(corrupted) == reference.decode(corrupted)
    assert ours.decode(corrupted)[0] == b"hello world"


def test_decode_ten_erasures_matches_upstream():
    ours, reference = mojo.RSCodec(10), upstream.RSCodec(10)
    encoded = ours.encode(b"erasures are located")
    positions = [0, 2, 4, 6, 8, 11, 14, 17, 20, 25]
    corrupted = bytearray(encoded)
    for pos in positions:
        corrupted[pos] = 0
    assert ours.decode(corrupted, erase_pos=positions) == reference.decode(
        corrupted, erase_pos=positions
    )


def test_mixed_errors_and_erasures_matches_upstream():
    ours, reference = mojo.RSCodec(12), upstream.RSCodec(12)
    encoded = ours.encode(bytes(range(100)))
    corrupted = bytearray(encoded)
    erasures = [2, 19, 47, 90]
    errors = [7, 33, 81, 105]
    for pos in erasures + errors:
        corrupted[pos] ^= 0xA7
    assert ours.decode(corrupted, erase_pos=erasures) == reference.decode(
        corrupted, erase_pos=erasures
    )


def test_uncorrectable_raises():
    codec = mojo.RSCodec(10)
    corrupted = codec.encode(b"hello world")
    for pos in range(6):
        corrupted[pos] ^= pos + 1
    with pytest.raises(mojo.ReedSolomonError):
        codec.decode(corrupted)


def test_long_message_chunking_parity():
    ours, reference = mojo.RSCodec(16), upstream.RSCodec(16)
    data = random.Random(42).randbytes(10_000)
    encoded = ours.encode(data)
    assert encoded == reference.encode(data)
    assert ours.check(encoded) == reference.check(encoded)
    assert ours.decode(encoded) == reference.decode(encoded)


def test_corruption_in_multiple_chunks():
    ours, reference = mojo.RSCodec(12), upstream.RSCodec(12)
    data = random.Random(9).randbytes(1000)
    corrupted = ours.encode(data)
    for pos in [10, 80, 260, 300, 520, 700, 900]:
        corrupted[pos] ^= 123
    assert ours.decode(corrupted) == reference.decode(corrupted)


def test_check_good_and_bad_parity():
    ours, reference = mojo.RSCodec(8, nsize=63), upstream.RSCodec(8, nsize=63)
    encoded = ours.encode(b"a" * 200)
    assert ours.check(encoded) == reference.check(encoded)
    encoded[70] ^= 1
    assert ours.check(encoded) == reference.check(encoded)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fcr": 1},
        {"fcr": 120, "prim": 0x187},
        {"generator": 3, "prim": 0x11B},
    ],
)
def test_custom_field_parameters(kwargs):
    ours = mojo.RSCodec(12, **kwargs)
    reference = upstream.RSCodec(12, **kwargs)
    data = bytes(range(120))
    encoded = ours.encode(data)
    assert encoded == reference.encode(data)
    encoded[11] ^= 73
    encoded[77] ^= 19
    assert ours.decode(encoded) == reference.decode(encoded)


def test_multiple_codec_instances_do_not_share_tables():
    standard = mojo.RSCodec(10)
    adsb = mojo.RSCodec(10, fcr=120, prim=0x187)
    data = b"independent codec tables"
    standard_encoded = standard.encode(data)
    adsb_encoded = adsb.encode(data)
    assert standard_encoded != adsb_encoded
    assert standard.decode(standard_encoded)[0] == data
    assert adsb.decode(adsb_encoded)[0] == data


def test_galois_scalar_helpers_match_upstream():
    reset()
    for x in range(1, 256, 13):
        for y in range(1, 256, 17):
            assert mojo.gf_add(x, y) == upstream.gf_add(x, y)
            assert mojo.gf_mul(x, y) == upstream.gf_mul(x, y)
            assert mojo.gf_div(x, y) == upstream.gf_div(x, y)
        assert mojo.gf_inverse(x) == upstream.gf_inverse(x)
        assert mojo.gf_pow(x, -17) == upstream.gf_pow(x, -17)


def test_polynomial_helpers_match_upstream():
    reset()
    p = bytearray([3, 0, 19, 27, 201])
    q = bytearray([1, 88, 0, 4])
    assert mojo.gf_poly_scale(p, 33) == upstream.gf_poly_scale(p, 33)
    assert mojo.gf_poly_add(p, q) == upstream.gf_poly_add(p, q)
    assert mojo.gf_poly_mul(p, q) == upstream.gf_poly_mul(p, q)
    assert mojo.gf_poly_div(p + bytearray(3), q) == upstream.gf_poly_div(
        p + bytearray(3), q
    )
    assert mojo.gf_poly_eval(p, 91) == upstream.gf_poly_eval(p, 91)
    assert mojo.gf_poly_mul_simple(p, q) == upstream.gf_poly_mul_simple(p, q)
    assert mojo.gf_poly_neg(p) == upstream.gf_poly_neg(p)
    assert mojo.gf_sub(91, 37) == upstream.gf_sub(91, 37)
    assert mojo.gf_neg(91) == upstream.gf_neg(91)


@pytest.mark.parametrize("p_len", [1, 3, 4, 5, 9, 13, 16, 17])
def test_simd_polynomial_multiply_tail_matches_upstream(p_len):
    reset()
    p = bytearray((i * 43 + 7) % 256 for i in range(p_len))
    q = bytearray([9, 0, 117, 3, 201])
    assert mojo.gf_poly_mul(p, q) == upstream.gf_poly_mul(p, q)


def test_generator_polynomials_match_upstream():
    reset()
    assert mojo.rs_generator_poly(32) == upstream.rs_generator_poly(32)
    assert mojo.rs_generator_poly_all(20) == upstream.rs_generator_poly_all(20)


@pytest.mark.parametrize("nsym", [0, 1, 7, 8, 9, 31, 32, 64])
def test_simd_generator_polynomial_tail_matches_upstream(nsym):
    reset()
    assert mojo.rs_generator_poly(nsym) == upstream.rs_generator_poly(nsym)


def test_prime_and_non_lut_helpers_match_upstream():
    for args in [(7, 11), (127, 83, 0x11D, 256, True)]:
        assert mojo.gf_mult_noLUT(*args) == upstream.gf_mult_noLUT(*args)
    assert mojo.gf_mult_noLUT_slow(127, 83, 0x11D) == upstream.gf_mult_noLUT_slow(
        127, 83, 0x11D
    )
    assert mojo.rwh_primes1(100) == upstream.rwh_primes1(100)
    assert mojo.find_prime_polys(single=True) == upstream.find_prime_polys(single=True)


def test_low_level_error_helpers_match_upstream():
    reset()
    codec = upstream.RSCodec(12)
    encoded = codec.encode(bytes(range(100)))
    for pos in [3, 41, 90]:
        encoded[pos] ^= 55
    synd = upstream.rs_calc_syndromes(encoded, 12)
    assert mojo.rs_forney_syndromes(synd, [], len(encoded)) == upstream.rs_forney_syndromes(
        synd, [], len(encoded)
    )
    ours_loc = mojo.rs_find_error_locator(synd, 12)
    ref_loc = upstream.rs_find_error_locator(synd, 12)
    assert ours_loc == ref_loc
    assert mojo.rs_find_errors(ours_loc[::-1], len(encoded)) == upstream.rs_find_errors(
        ref_loc[::-1], len(encoded)
    )
    positions = [3, 41, 90]
    coef_positions = [len(encoded) - 1 - pos for pos in positions]
    assert mojo.rs_find_errata_locator(
        coef_positions
    ) == upstream.rs_find_errata_locator(coef_positions)
    assert mojo.rs_find_error_evaluator(
        synd[::-1], ours_loc, len(ours_loc) - 1
    ) == upstream.rs_find_error_evaluator(
        synd[::-1], ref_loc, len(ref_loc) - 1
    )
    assert mojo.rs_correct_errata(encoded, synd, positions) == upstream.rs_correct_errata(
        encoded, synd, positions
    )
    assert mojo.rs_check(encoded, 12) == upstream.rs_check(encoded, 12)
    assert mojo.rs_simple_encode_msg(b"covered", 12) == upstream.rs_simple_encode_msg(
        b"covered", 12
    )


@pytest.mark.parametrize("nsize", [63, 254, 255])
def test_simd_chien_search_tail_matches_upstream(nsize):
    reset()
    nsym = 12
    data = bytes((i * 29 + 3) % 256 for i in range(nsize - nsym))
    encoded = upstream.rs_encode_msg(data, nsym)
    for pos in [1, nsize // 3, nsize - 2]:
        encoded[pos] ^= 0x6D
    synd = upstream.rs_calc_syndromes(encoded, nsym)
    locator = upstream.rs_find_error_locator(synd, nsym)[::-1]
    assert mojo.rs_find_errors(locator, nsize) == upstream.rs_find_errors(
        locator, nsize
    )


def test_chien_search_rejects_malformed_locator():
    reset()
    with pytest.raises(mojo.ReedSolomonError):
        mojo.rs_find_errors([0, 0], 255)


def test_ffi_inputs_reject_silent_narrowing_and_bad_shapes():
    reset()
    with pytest.raises(ValueError, match="range"):
        mojo.rs_encode_msg([256], 2)
    with pytest.raises(ValueError, match="range"):
        mojo.gf_poly_mul([-1], [1])
    with pytest.raises(ValueError, match="one-dimensional"):
        mojo.gf_poly_mul(np.ones((2, 2), dtype=np.uint8), [1])
    with pytest.raises(ValueError, match="contiguous"):
        mojo.rs_encode_msg(memoryview(bytearray(range(8)))[::2], 2)
    assert mojo.rs_encode_msg(np.array([1, 2], dtype=np.uint16), 2) == (
        upstream.rs_encode_msg(bytearray([1, 2]), 2)
    )
    assert mojo.gf_poly_mul([], [1]) == bytearray()


def test_ffi_exports_reject_null_pointers():
    native = mojo.reedsolo.lib()
    assert native.mrs_encode(0, 1, 2, 0, 0, 0, 0) < 0
    assert native.mrs_syndromes(0, 1, 2, 0, 2, 0, 0, 0) < 0
    assert native.mrs_poly_mul(0, 1, 0, 1, 0, 0, 0) < 0
    assert native.mrs_generator_poly(1, 0, 2, 0, 0, 0) < 0
    assert native.mrs_find_errors(0, 1, 1, 2, 0, 0, 0) < 0


def test_nofsynd_decoder_and_square_match_upstream():
    reset()
    encoded = upstream.RSCodec(10).encode(b"hello world")
    erasures = [1, 5]
    for pos in erasures + [8]:
        encoded[pos] ^= 55
    assert mojo.rs_correct_msg_nofsynd(
        encoded, 10, erase_pos=erasures
    ) == upstream.rs_correct_msg_nofsynd(encoded, 10, erase_pos=erasures)
    for poly in ([0], [1, 2, 3], [5, 0, 9, 11]):
        assert mojo.gf_poly_square(poly) == upstream.gf_poly_square(poly)


def test_maxerrata_parity():
    ours, reference = mojo.RSCodec(13), upstream.RSCodec(13)
    for errors, erasures in [(None, None), (3, None), (None, 5)]:
        assert ours.maxerrata(errors, erasures) == reference.maxerrata(
            errors, erasures
        )


def test_public_signatures_match_upstream():
    names = [
        "init_tables",
        "rs_encode_msg",
        "rs_calc_syndromes",
        "rs_correct_msg",
        "rs_check",
        "RSCodec",
    ]
    for name in names:
        assert inspect.signature(getattr(mojo, name)) == inspect.signature(
            getattr(upstream, name)
        )
