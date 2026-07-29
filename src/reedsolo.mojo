from std.sys import simd_width_of


comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def gf_pow(log: BPtr, exp: BPtr, x: Int, power: Int) -> Int:
    var index = (Int(log[x]) * power) % 255
    if index < 0:
        index += 255
    return Int(exp[index])


def poly_eval(log: BPtr, exp: BPtr, poly: BPtr, n: Int, x: Int) -> Int:
    var y = Int(poly[0])
    for i in range(1, n):
        if y == 0 or x == 0:
            y = Int(poly[i])
        else:
            y = Int(exp[Int(log[y]) + Int(log[x])]) ^ Int(poly[i])
    return y


def find_errors_simd(
    err_loc: BPtr,
    err_len: Int,
    nmess: Int,
    generator: Int,
    log: BPtr,
    exp: BPtr,
    dst: BPtr,
) -> Int:
    comptime W = simd_width_of[DType.float64]()
    var count = 0
    var i = 0
    while i + W <= nmess:
        var xs = SIMD[DType.uint8, W]()
        for lane in range(W):
            xs[lane] = UInt8(gf_pow(log, exp, generator, i + lane))
        var ys = SIMD[DType.uint8, W](err_loc[0])
        var log_x = log.gather[width=W](xs.cast[DType.int64]())
        for j in range(1, err_len):
            var nonzero = ys.ne(SIMD[DType.uint8, W](0))
            var log_y = log.gather[width=W](
                ys.cast[DType.int64](), mask=nonzero
            )
            var product = exp.gather[width=W](
                log_y.cast[DType.int64]() + log_x.cast[DType.int64](),
                mask=nonzero,
            )
            ys = product ^ SIMD[DType.uint8, W](err_loc[j])
        for lane in range(W):
            if ys[lane] == 0:
                dst[count] = UInt8(nmess - 1 - i - lane)
                count += 1
        i += W
    while i < nmess:
        var x = gf_pow(log, exp, generator, i)
        if poly_eval(log, exp, err_loc, err_len, x) == 0:
            dst[count] = UInt8(nmess - 1 - i)
            count += 1
        i += 1
    return count


@export("mrs_encode")
def mrs_encode(
    msg_addr: Int,
    msg_len: Int,
    nsym: Int,
    gen_addr: Int,
    log_addr: Int,
    exp_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        msg_addr == 0
        or gen_addr == 0
        or log_addr == 0
        or exp_addr == 0
        or dst_addr == 0
        or msg_len < 0
        or nsym <= 0
        or msg_len + nsym > 255
    ):
        return -1
    var msg = BPtr(unsafe_from_address=msg_addr)
    var gen = BPtr(unsafe_from_address=gen_addr)
    var log = BPtr(unsafe_from_address=log_addr)
    var exp = BPtr(unsafe_from_address=exp_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    for i in range(msg_len):
        dst[i] = msg[i]
    for i in range(nsym):
        dst[msg_len + i] = 0
    for i in range(msg_len):
        var coef = Int(dst[i])
        if coef != 0:
            var lcoef = Int(log[coef])
            for j in range(1, nsym + 1):
                dst[i + j] = dst[i + j] ^ exp[lcoef + Int(log[Int(gen[j])])]
    for i in range(msg_len):
        dst[i] = msg[i]
    return msg_len + nsym


@export("mrs_syndromes")
def mrs_syndromes(
    msg_addr: Int,
    msg_len: Int,
    nsym: Int,
    fcr: Int,
    generator: Int,
    log_addr: Int,
    exp_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        msg_addr == 0
        or log_addr == 0
        or exp_addr == 0
        or dst_addr == 0
        or msg_len <= 0
        or msg_len > 255
        or nsym <= 0
        or nsym >= 255
    ):
        return -1
    var msg = BPtr(unsafe_from_address=msg_addr)
    var log = BPtr(unsafe_from_address=log_addr)
    var exp = BPtr(unsafe_from_address=exp_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var nonzero = 0
    for i in range(nsym):
        var x = gf_pow(log, exp, generator, i + fcr)
        var value = poly_eval(log, exp, msg, msg_len, x)
        dst[i] = UInt8(value)
        nonzero |= value
    return nonzero


@export("mrs_poly_mul")
def mrs_poly_mul(
    p_addr: Int,
    p_len: Int,
    q_addr: Int,
    q_len: Int,
    log_addr: Int,
    exp_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        p_addr == 0
        or q_addr == 0
        or log_addr == 0
        or exp_addr == 0
        or dst_addr == 0
        or p_len <= 0
        or q_len <= 0
    ):
        return -1
    var p = BPtr(unsafe_from_address=p_addr)
    var q = BPtr(unsafe_from_address=q_addr)
    var log = BPtr(unsafe_from_address=log_addr)
    var exp = BPtr(unsafe_from_address=exp_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    var n = p_len + q_len - 1
    for i in range(n):
        dst[i] = 0
    for j in range(q_len):
        var qv = Int(q[j])
        if qv != 0:
            var lq = Int(log[qv])
            comptime W = simd_width_of[DType.float64]()
            var i = 0
            while i + W <= p_len:
                var pv = p.load[width=W](i)
                var nonzero = pv.ne(SIMD[DType.uint8, W](0))
                var lp = log.gather[width=W](
                    pv.cast[DType.int64](), mask=nonzero
                )
                var product = exp.gather[width=W](
                    lp.cast[DType.int64]() + SIMD[DType.int64, W](lq),
                    mask=nonzero,
                )
                dst.store(i + j, dst.load[width=W](i + j) ^ product)
                i += W
            while i < p_len:
                var pv = Int(p[i])
                if pv != 0:
                    dst[i + j] = dst[i + j] ^ exp[Int(log[pv]) + lq]
                i += 1
    return n


@export("mrs_find_errors")
def mrs_find_errors(
    err_loc_addr: Int,
    err_len: Int,
    nmess: Int,
    generator: Int,
    log_addr: Int,
    exp_addr: Int,
    dst_addr: Int,
) abi("C") -> Int:
    if (
        err_loc_addr == 0
        or log_addr == 0
        or exp_addr == 0
        or dst_addr == 0
        or err_len <= 0
        or nmess <= 0
        or nmess > 255
    ):
        return -1
    var err_loc = BPtr(unsafe_from_address=err_loc_addr)
    var log = BPtr(unsafe_from_address=log_addr)
    var exp = BPtr(unsafe_from_address=exp_addr)
    var dst = BPtr(unsafe_from_address=dst_addr)
    return find_errors_simd(
        err_loc, err_len, nmess, generator, log, exp, dst
    )
