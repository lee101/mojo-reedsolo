from __future__ import annotations

import itertools
import math
from array import array

import numpy as np

from ._lib import lib


class ReedSolomonError(Exception):
    pass


field_charac = 255
gf_exp = bytearray([1] * 510)
gf_log = bytearray(256)
_bytearray = bytearray
xrange = range


def gf_mult_noLUT(x, y, prim=0, field_charac_full=256, carryless=True):
    result = 0
    while y:
        if y & 1:
            result = result ^ x if carryless else result + x
        y >>= 1
        x <<= 1
        if prim > 0 and x & field_charac_full:
            x ^= prim
    return result


def gf_mult_noLUT_slow(x, y, prim=0):
    result = 0
    while y:
        if y & 1:
            result ^= x
        x <<= 1
        y >>= 1
    if prim:
        top = prim.bit_length()
        while result.bit_length() >= top:
            result ^= prim << (result.bit_length() - top)
    return result


def rwh_primes1(n):
    sieve = [True] * (n // 2)
    for i in range(3, int(n**0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[(i * i) // 2 :: i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
    return [2] + [2 * i + 1 for i in range(1, n // 2) if sieve[i]]


def find_prime_polys(generator=2, c_exp=8, fast_primes=False, single=False):
    characteristic = 2**c_exp - 1
    upper = 2 ** (c_exp + 1) - 1
    candidates = (
        [x for x in rwh_primes1(upper) if x > characteristic]
        if fast_primes
        else range(characteristic + 2, upper, 2)
    )
    found = []
    for prim in candidates:
        seen = bytearray(characteristic + 1)
        x = 1
        for _ in range(characteristic):
            x = gf_mult_noLUT(x, generator, prim, characteristic + 1)
            if x > characteristic or seen[x]:
                break
            seen[x] = 1
        else:
            if single:
                return prim
            found.append(prim)
    return found


def init_tables(prim=0x11D, generator=2, c_exp=8):
    global gf_exp, gf_log, field_charac, _bytearray
    if c_exp != 8:
        raise NotImplementedError("mojo-reedsolo currently supports GF(2^8) symbols")
    field_charac = 2**c_exp - 1
    _bytearray = bytearray if c_exp <= 8 else lambda obj=0: array(
        "i", [0] * obj if isinstance(obj, int) else obj
    )
    gf_exp = _bytearray(field_charac * 2)
    gf_log = _bytearray(field_charac + 1)
    x = 1
    for i in range(field_charac):
        gf_exp[i] = x
        gf_log[x] = i
        x = gf_mult_noLUT(x, generator, prim, field_charac + 1)
    for i in range(field_charac, field_charac * 2):
        gf_exp[i] = gf_exp[i - field_charac]
    return [gf_log, gf_exp, field_charac]


def gf_add(x, y):
    return x ^ y


def gf_sub(x, y):
    return x ^ y


def gf_neg(x):
    return x


def gf_inverse(x):
    return gf_exp[field_charac - gf_log[x]]


def gf_mul(x, y):
    if x == 0 or y == 0:
        return 0
    return gf_exp[(gf_log[x] + gf_log[y]) % field_charac]


def gf_div(x, y):
    if y == 0:
        raise ZeroDivisionError
    if x == 0:
        return 0
    return gf_exp[(gf_log[x] + field_charac - gf_log[y]) % field_charac]


def gf_pow(x, power):
    return gf_exp[(gf_log[x] * power) % field_charac]


def gf_poly_scale(p, x):
    return _bytearray(gf_mul(v, x) for v in p)


def gf_poly_add(p, q):
    result = _bytearray(max(len(p), len(q)))
    result[len(result) - len(p) :] = p
    offset = len(result) - len(q)
    for i, value in enumerate(q):
        result[i + offset] ^= value
    return result


def gf_poly_mul(p, q):
    if field_charac == 255:
        p_buffer = isinstance(p, (bytes, bytearray))
        q_buffer = isinstance(q, (bytes, bytearray))
        p_len = len(p) if p_buffer else len(_u8_table(p, "p"))
        q_len = len(q) if q_buffer else len(_u8_table(q, "q"))
        if not p_len or not q_len:
            return bytearray()
        if p_len * q_len <= 64:
            pv = p if p_buffer else bytearray(_u8_table(p, "p"))
            qv = q if q_buffer else bytearray(_u8_table(q, "q"))
            result = bytearray(p_len + q_len - 1)
            log_p = [gf_log[value] for value in pv]
            for j, value_q in enumerate(qv):
                if value_q:
                    log_q = gf_log[value_q]
                    for i, value_p in enumerate(pv):
                        if value_p:
                            result[i + j] ^= gf_exp[log_p[i] + log_q]
            return result
        pa = np.frombuffer(p, dtype=np.uint8) if p_buffer else _u8_table(p, "p")
        qa = np.frombuffer(q, dtype=np.uint8) if q_buffer else _u8_table(q, "q")
        logs = np.frombuffer(gf_log, dtype=np.uint8)
        exps = np.frombuffer(gf_exp, dtype=np.uint8)
        result = bytearray(len(pa) + len(qa) - 1)
        dst = np.frombuffer(result, dtype=np.uint8)
        status = lib().mrs_poly_mul(
            pa.ctypes.data,
            len(pa),
            qa.ctypes.data,
            len(qa),
            logs.ctypes.data,
            exps.ctypes.data,
            dst.ctypes.data,
        )
        if status != len(dst):
            raise RuntimeError("Mojo polynomial multiplication failed")
        return result
    result = _bytearray(len(p) + len(q) - 1)
    for j, qv in enumerate(q):
        if qv:
            for i, pv in enumerate(p):
                if pv:
                    result[i + j] ^= gf_mul(pv, qv)
    return result


def gf_poly_mul_simple(p, q):
    return gf_poly_mul(p, q)


def gf_poly_neg(poly):
    return poly


def gf_poly_div(dividend, divisor):
    result = _bytearray(dividend)
    for i in range(len(dividend) - len(divisor) + 1):
        coef = result[i]
        if coef:
            for j in range(1, len(divisor)):
                if divisor[j]:
                    result[i + j] ^= gf_mul(divisor[j], coef)
    separator = len(result) - len(divisor) + 1
    return result[:separator], result[separator:]


def gf_poly_square(poly):
    result = _bytearray(2 * len(poly) - 1)
    for i in range(len(poly) - 1):
        if poly[i]:
            result[2 * i] = gf_exp[2 * gf_log[poly[i]]]
    result[-1] = gf_exp[2 * gf_log[poly[-1]]]
    if result[0] == 0:
        result[0] = 2 * poly[1] - 1
    return result


def gf_poly_eval(poly, x):
    y = poly[0]
    for value in poly[1:]:
        y = gf_mul(y, x) ^ value
    return y


def rs_generator_poly(nsym, fcr=0, generator=2):
    if field_charac == 255 and 0 <= nsym < 255:
        result = bytearray(nsym + 1)
        dst = np.frombuffer(result, dtype=np.uint8)
        logs = np.frombuffer(gf_log, dtype=np.uint8)
        exps = np.frombuffer(gf_exp, dtype=np.uint8)
        status = lib().mrs_generator_poly(
            nsym,
            fcr,
            generator,
            logs.ctypes.data,
            exps.ctypes.data,
            dst.ctypes.data,
        )
        if status != len(result):
            raise ValueError("invalid Reed-Solomon generator parameters")
        return result
    result = _bytearray([1])
    for i in range(nsym):
        result = gf_poly_mul(result, [1, gf_pow(generator, i + fcr)])
    return result


def rs_generator_poly_all(max_nsym, fcr=0, generator=2):
    return {i: rs_generator_poly(i, fcr, generator) for i in range(max_nsym)}


def _u8_table(values, name="values"):
    """Return a contiguous one-dimensional GF(256) view without narrowing."""
    if isinstance(values, (bytes, bytearray)):
        return np.frombuffer(values, dtype=np.uint8)
    if isinstance(values, memoryview):
        if values.ndim != 1 or values.itemsize != 1 or not values.contiguous:
            raise ValueError(f"{name} must be a contiguous one-byte symbol buffer")
        return np.frombuffer(values, dtype=np.uint8)

    source = np.asarray(values)
    if source.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not source.size:
        return np.empty(0, dtype=np.uint8)
    if source.dtype.kind not in "biu":
        raise TypeError(f"{name} must contain integer GF(256) symbols")
    if source.size and (np.any(source < 0) or np.any(source > 255)):
        raise ValueError(f"{name} symbols must be in range(0, 256)")
    return np.ascontiguousarray(source, dtype=np.uint8)


def rs_encode_msg(msg_in, nsym, fcr=0, generator=2, gen=None):
    if len(msg_in) + nsym > field_charac:
        raise ValueError(
            f"Message is too long ({len(msg_in) + nsym} when max is {field_charac})"
        )
    if gen is None:
        gen = rs_generator_poly(nsym, fcr, generator)
    if field_charac != 255:
        _, remainder = gf_poly_div(
            _bytearray(msg_in) + _bytearray(nsym), gen
        )
        return _bytearray(msg_in) + remainder
    msg = _u8_table(msg_in, "msg_in")
    if nsym == 0:
        return bytearray(msg)
    generator_poly = _u8_table(gen, "gen")
    if len(generator_poly) < nsym + 1:
        raise ValueError("gen must contain at least nsym + 1 coefficients")
    logs = _u8_table(gf_log)
    exps = _u8_table(gf_exp)
    result = bytearray(len(msg) + nsym)
    dst = np.frombuffer(result, dtype=np.uint8)
    status = lib().mrs_encode(
        msg.ctypes.data,
        len(msg),
        nsym,
        generator_poly.ctypes.data,
        logs.ctypes.data,
        exps.ctypes.data,
        dst.ctypes.data,
    )
    if status < 0:
        raise ValueError("invalid Reed-Solomon encoding parameters")
    return result


def rs_simple_encode_msg(msg_in, nsym, fcr=0, generator=2, gen=None):
    return rs_encode_msg(msg_in, nsym, fcr, generator, gen)


def rs_calc_syndromes(msg, nsym, fcr=0, generator=2):
    if field_charac == 255 and len(msg):
        source = _u8_table(msg, "msg")
        logs = _u8_table(gf_log)
        exps = _u8_table(gf_exp)
        dst = np.empty(nsym, dtype=np.uint8)
        status = lib().mrs_syndromes(
            source.ctypes.data,
            len(source),
            nsym,
            fcr,
            generator,
            logs.ctypes.data,
            exps.ctypes.data,
            dst.ctypes.data,
        )
        if status < 0:
            raise ValueError("invalid Reed-Solomon syndrome parameters")
        return [0, *dst.tolist()]
    return [0] + [
        gf_poly_eval(msg, gf_pow(generator, i + fcr)) for i in range(nsym)
    ]


def rs_check(msg, nsym, fcr=0, generator=2):
    return max(rs_calc_syndromes(msg, nsym, fcr, generator)) == 0


def rs_find_errata_locator(e_pos, generator=2):
    result = _bytearray([1])
    for pos in e_pos:
        result = gf_poly_mul(
            result, gf_poly_add(_bytearray([1]), [gf_pow(generator, pos), 0])
        )
    return result


def rs_find_error_evaluator(synd, err_loc, nsym):
    product = gf_poly_mul(synd, err_loc)
    return product[-(nsym + 1) :]


def rs_find_error_locator(synd, nsym, erase_loc=None, erase_count=0):
    err_loc = _bytearray(erase_loc) if erase_loc else _bytearray([1])
    old_loc = _bytearray(err_loc)
    synd_shift = max(0, len(synd) - nsym)
    for i in range(nsym - erase_count):
        k = i + synd_shift + (erase_count if erase_loc else 0)
        delta = synd[k]
        for j in range(1, len(err_loc)):
            delta ^= gf_mul(err_loc[-(j + 1)], synd[k - j])
        old_loc.append(0)
        if delta:
            if len(old_loc) > len(err_loc):
                new_loc = gf_poly_scale(old_loc, delta)
                old_loc = gf_poly_scale(err_loc, gf_inverse(delta))
                err_loc = new_loc
            err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))
    err_loc = list(itertools.dropwhile(lambda x: x == 0, err_loc))
    errs = len(err_loc) - 1
    if (errs - erase_count) * 2 + erase_count > nsym:
        raise ReedSolomonError("Too many errors to correct")
    return err_loc


def rs_find_errors(err_loc, nmess, generator=2):
    expected = len(err_loc) - 1
    if field_charac == 255 and nmess:
        locator = _u8_table(err_loc, "err_loc")
        logs = _u8_table(gf_log)
        exps = _u8_table(gf_exp)
        found = np.empty(nmess, dtype=np.uint8)
        count = lib().mrs_find_errors(
            locator.ctypes.data,
            len(locator),
            nmess,
            generator,
            logs.ctypes.data,
            exps.ctypes.data,
            found.ctypes.data,
        )
        if count < 0:
            raise ValueError("invalid Reed-Solomon Chien search parameters")
        positions = found[:count].tolist()
    else:
        positions = [
            nmess - 1 - i
            for i in range(nmess)
            if gf_poly_eval(err_loc, gf_pow(generator, i)) == 0
        ]
    if len(positions) != expected:
        raise ReedSolomonError(
            "Too many (or few) errors found by Chien Search for the errata locator polynomial!"
        )
    return positions


def rs_forney_syndromes(synd, pos, nmess, generator=2):
    result = list(synd[1:])
    for position in pos:
        x = gf_pow(generator, nmess - 1 - position)
        for j in range(len(result) - 1):
            result[j] = gf_mul(result[j], x) ^ result[j + 1]
    return result


def rs_correct_errata(msg_in, synd, err_pos, fcr=0, generator=2):
    msg = _bytearray(msg_in)
    coef_pos = [len(msg) - 1 - pos for pos in err_pos]
    err_loc = rs_find_errata_locator(coef_pos, generator)
    err_eval = rs_find_error_evaluator(
        list(reversed(synd)), err_loc, len(err_loc) - 1
    )
    err_eval.reverse()
    locations = [
        gf_pow(generator, -(field_charac - position)) for position in coef_pos
    ]
    for i, xi in enumerate(locations):
        xi_inv = gf_inverse(xi)
        prime = 1
        for j, other in enumerate(locations):
            if j != i:
                prime = gf_mul(prime, gf_sub(1, gf_mul(xi_inv, other)))
        if prime == 0:
            raise ReedSolomonError("Forney algorithm could not locate errors")
        y = gf_poly_eval(list(reversed(err_eval)), xi_inv)
        y = gf_mul(gf_pow(xi, 1 - fcr), y)
        msg[err_pos[i]] ^= gf_div(y, prime)
    return msg


def rs_correct_msg(
    msg_in,
    nsym,
    fcr=0,
    generator=2,
    erase_pos=None,
    only_erasures=False,
):
    if len(msg_in) > field_charac:
        raise ValueError(f"Message is too long ({len(msg_in)} when max is {field_charac})")
    msg = _bytearray(msg_in)
    erasures = [] if erase_pos is None else list(erase_pos)
    for position in erasures:
        msg[position] = 0
    if len(erasures) > nsym:
        raise ReedSolomonError("Too many erasures to correct")
    synd = rs_calc_syndromes(msg, nsym, fcr, generator)
    if max(synd) == 0:
        return msg[:-nsym], msg[-nsym:], erasures
    errors = []
    if not only_erasures:
        fsynd = rs_forney_syndromes(synd, erasures, len(msg), generator)
        err_loc = rs_find_error_locator(
            fsynd, nsym, erase_count=len(erasures)
        )
        errors = rs_find_errors(list(reversed(err_loc)), len(msg), generator)
    msg = rs_correct_errata(
        msg, synd, erasures + errors, fcr=fcr, generator=generator
    )
    if max(rs_calc_syndromes(msg, nsym, fcr, generator)):
        raise ReedSolomonError("Could not correct message")
    return msg[:-nsym], msg[-nsym:], erasures + errors


def rs_correct_msg_nofsynd(
    msg_in,
    nsym,
    fcr=0,
    generator=2,
    erase_pos=None,
    only_erasures=False,
):
    if len(msg_in) > field_charac:
        raise ValueError(f"Message is too long ({len(msg_in)} when max is {field_charac})")
    msg = _bytearray(msg_in)
    erasures = [] if erase_pos is None else list(erase_pos)
    for position in erasures:
        msg[position] = 0
    if len(erasures) > nsym:
        raise ReedSolomonError("Too many erasures to correct")
    synd = rs_calc_syndromes(msg, nsym, fcr, generator)
    if max(synd) == 0:
        return msg[:-nsym], msg[-nsym:], []
    erase_loc = None
    if erasures:
        erase_loc = rs_find_errata_locator(
            [len(msg) - 1 - position for position in erasures], generator
        )
    if only_erasures:
        err_loc = list(reversed(erase_loc))
    else:
        err_loc = list(
            reversed(
                rs_find_error_locator(
                    synd,
                    nsym,
                    erase_loc=erase_loc,
                    erase_count=len(erasures),
                )
            )
        )
    errors = rs_find_errors(err_loc, len(msg), generator)
    msg = rs_correct_errata(msg, synd, errors, fcr, generator)
    if max(rs_calc_syndromes(msg, nsym, fcr, generator)):
        raise ReedSolomonError("Could not correct message")
    return msg[:-nsym], msg[-nsym:], erasures + errors


class RSCodec:
    def __init__(
        self,
        nsym=10,
        nsize=255,
        fcr=0,
        prim=0x11D,
        generator=2,
        c_exp=8,
        single_gen=True,
    ):
        if nsize > 255 and c_exp <= 8:
            c_exp = math.ceil(math.log2(nsize + 1))
        if c_exp != 8 and prim == 0x11D:
            prim = find_prime_polys(generator, c_exp, fast_primes=True, single=True)
            if nsize == 255:
                nsize = 2**c_exp - 1
        if nsym >= nsize:
            raise ValueError(
                "ECC symbols must be strictly less than the total message length (nsym < nsize)."
            )
        self.nsym = nsym
        self.nsize = nsize
        self.fcr = fcr
        self.prim = prim
        self.generator = generator
        self.c_exp = c_exp
        self.gf_log, self.gf_exp, self.field_charac = init_tables(
            prim, generator, c_exp
        )
        self.gen = (
            {nsym: rs_generator_poly(nsym, fcr, generator)}
            if single_gen
            else rs_generator_poly_all(nsize, fcr, generator)
        )

    def _restore(self):
        global gf_log, gf_exp, field_charac, _bytearray
        gf_log, gf_exp, field_charac = (
            self.gf_log,
            self.gf_exp,
            self.field_charac,
        )
        _bytearray = bytearray if self.c_exp <= 8 else lambda obj=0: array(
            "i", [0] * obj if isinstance(obj, int) else obj
        )

    def chunk(self, data, chunksize):
        for i in range(0, len(data), chunksize):
            yield data[i : i + chunksize]

    def encode(self, data, nsym=None):
        self._restore()
        nsym = nsym or self.nsym
        encoded = _bytearray()
        for chunk in self.chunk(data, self.nsize - self.nsym):
            gen = self.gen.get(nsym) or rs_generator_poly(
                nsym, self.fcr, self.generator
            )
            encoded.extend(
                rs_encode_msg(chunk, nsym, self.fcr, self.generator, gen)
            )
        return encoded

    def decode(self, data, nsym=None, erase_pos=None, only_erasures=False):
        self._restore()
        nsym = nsym or self.nsym
        decoded = _bytearray()
        corrected = _bytearray()
        all_positions = _bytearray()
        remaining = list(erase_pos or [])
        for chunk in self.chunk(data, self.nsize):
            positions = [x for x in remaining if x < self.nsize]
            remaining = [x - self.nsize for x in remaining if x >= self.nsize]
            msg, ecc, errata = rs_correct_msg(
                chunk,
                nsym,
                self.fcr,
                self.generator,
                positions,
                only_erasures,
            )
            decoded.extend(msg)
            corrected.extend(msg + ecc)
            all_positions.extend(errata)
        return decoded, corrected, all_positions

    def check(self, data, nsym=None):
        self._restore()
        nsym = nsym or self.nsym
        return [
            rs_check(chunk, nsym, self.fcr, self.generator)
            for chunk in self.chunk(data, self.nsize)
        ]

    def maxerrata(self, errors=None, erasures=None, verbose=False):
        maxerrors, maxerasures = self.nsym // 2, self.nsym
        if erasures is not None and erasures >= 0:
            if erasures > maxerasures:
                raise ReedSolomonError(
                    "Specified number of errors or erasures exceeding the Singleton Bound!"
                )
            result = (self.nsym - erasures) // 2, erasures
            if verbose:
                print(
                    f"This codec can correct up to {result[0]} errors and "
                    f"{result[1]} erasures simultaneously"
                )
            return result
        if errors is not None and errors >= 0:
            if errors > maxerrors:
                raise ReedSolomonError(
                    "Specified number of errors or erasures exceeding the Singleton Bound!"
                )
            result = errors, self.nsym - errors * 2
            if verbose:
                print(
                    f"This codec can correct up to {result[0]} errors and "
                    f"{result[1]} erasures simultaneously"
                )
            return result
        if verbose:
            print(
                f"This codec can correct up to {maxerrors} errors and "
                f"{maxerasures} erasures independently"
            )
        return maxerrors, maxerasures


init_tables()
