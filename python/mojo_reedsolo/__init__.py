from .reedsolo import (
    RSCodec,
    ReedSolomonError,
    find_prime_polys,
    gf_add,
    gf_div,
    gf_inverse,
    gf_mul,
    gf_mult_noLUT,
    gf_mult_noLUT_slow,
    gf_neg,
    gf_poly_add,
    gf_poly_div,
    gf_poly_eval,
    gf_poly_mul,
    gf_poly_mul_simple,
    gf_poly_neg,
    gf_poly_scale,
    gf_poly_square,
    gf_pow,
    gf_sub,
    init_tables,
    rs_calc_syndromes,
    rs_check,
    rs_correct_errata,
    rs_correct_msg,
    rs_correct_msg_nofsynd,
    rs_encode_msg,
    rs_find_errata_locator,
    rs_find_error_evaluator,
    rs_find_error_locator,
    rs_find_errors,
    rs_forney_syndromes,
    rs_generator_poly,
    rs_generator_poly_all,
    rs_simple_encode_msg,
    rwh_primes1,
    xrange,
)

__all__ = [name for name in globals() if not name.startswith("_")] + [
    "gf_log",
    "gf_exp",
    "field_charac",
]


def __getattr__(name):
    if name in {"gf_log", "gf_exp", "field_charac"}:
        from . import reedsolo

        return getattr(reedsolo, name)
    raise AttributeError(name)
