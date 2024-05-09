use core::arch::x86_64::*;

use super::goldilocks_avx2::{reduce_avx_96_64, sqr_avx};
use super::utils_avx2::{add64, add64_no_carry, mul64_no_overflow};
use crate::monolith_hash::monolith_goldilocks::monolith_mds_12::{block2, MDS_FREQ_BLOCK_ONE, MDS_FREQ_BLOCK_THREE, MDS_FREQ_BLOCK_TWO};
use crate::monolith_hash::monolith_goldilocks::{MONOLITH_MAT_12, MONOLITH_ROUND_CONSTANTS};
use crate::monolith_hash::{LOOKUP_BITS, SPONGE_WIDTH};

#[inline]
unsafe fn bar_avx(el: &mut __m256i) {
    if LOOKUP_BITS == 8 {
        let ct1 = _mm256_set_epi64x(
            0x8080808080808080u64 as i64,
            0x8080808080808080u64 as i64,
            0x8080808080808080u64 as i64,
            0x8080808080808080u64 as i64,
        );
        let ct2 = _mm256_set_epi64x(
            0x7F7F7F7F7F7F7F7Fu64 as i64,
            0x7F7F7F7F7F7F7F7Fu64 as i64,
            0x7F7F7F7F7F7F7F7Fu64 as i64,
            0x7F7F7F7F7F7F7F7Fu64 as i64,
        );
        let ct3 = _mm256_set_epi64x(
            0xC0C0C0C0C0C0C0C0u64 as i64,
            0xC0C0C0C0C0C0C0C0u64 as i64,
            0xC0C0C0C0C0C0C0C0u64 as i64,
            0xC0C0C0C0C0C0C0C0u64 as i64,
        );
        let ct4 = _mm256_set_epi64x(
            0x3F3F3F3F3F3F3F3Fu64 as i64,
            0x3F3F3F3F3F3F3F3Fu64 as i64,
            0x3F3F3F3F3F3F3F3Fu64 as i64,
            0x3F3F3F3F3F3F3F3Fu64 as i64,
        );
        let ct5 = _mm256_set_epi64x(
            0xE0E0E0E0E0E0E0E0u64 as i64,
            0xE0E0E0E0E0E0E0E0u64 as i64,
            0xE0E0E0E0E0E0E0E0u64 as i64,
            0xE0E0E0E0E0E0E0E0u64 as i64,
        );
        let ct6 = _mm256_set_epi64x(
            0x1F1F1F1F1F1F1F1Fu64 as i64,
            0x1F1F1F1F1F1F1F1Fu64 as i64,
            0x1F1F1F1F1F1F1F1Fu64 as i64,
            0x1F1F1F1F1F1F1F1Fu64 as i64,
        );
        let l1 = _mm256_andnot_si256(*el, ct1);
        let l2 = _mm256_srli_epi64(l1, 7);
        let l3 = _mm256_andnot_si256(*el, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        let limb1 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct3);
        let l2 = _mm256_srli_epi64(l1, 6);
        let l3 = _mm256_and_si256(*el, ct4);
        let l4 = _mm256_slli_epi64(l3, 2);
        let limb2 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct5);
        let l2 = _mm256_srli_epi64(l1, 5);
        let l3 = _mm256_and_si256(*el, ct6);
        let l4 = _mm256_slli_epi64(l3, 3);
        let limb3 = _mm256_or_si256(l2, l4);
        let tmp = _mm256_and_si256(limb1, limb2);
        let tmp = _mm256_and_si256(tmp, limb3);
        let tmp = _mm256_xor_si256(*el, tmp);
        let l1 = _mm256_and_si256(tmp, ct1);
        let l2 = _mm256_srli_epi64(l1, 7);
        let l3 = _mm256_and_si256(tmp, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        *el = _mm256_or_si256(l2, l4);
    } else if LOOKUP_BITS == 16 {
        let ct1 = _mm256_set_epi64x(
            0x8000800080008000u64 as i64,
            0x8000800080008000u64 as i64,
            0x8000800080008000u64 as i64,
            0x8000800080008000u64 as i64,
        );
        let ct2 = _mm256_set_epi64x(
            0x7FFF7FFF7FFF7FFFu64 as i64,
            0x7FFF7FFF7FFF7FFFu64 as i64,
            0x7FFF7FFF7FFF7FFFu64 as i64,
            0x7FFF7FFF7FFF7FFFu64 as i64,
        );
        let ct3 = _mm256_set_epi64x(
            0xC000C000C000C000u64 as i64,
            0xC000C000C000C000u64 as i64,
            0xC000C000C000C000u64 as i64,
            0xC000C000C000C000u64 as i64,
        );
        let ct4 = _mm256_set_epi64x(
            0x3FFF3FFF3FFF3FFFu64 as i64,
            0x3FFF3FFF3FFF3FFFu64 as i64,
            0x3FFF3FFF3FFF3FFFu64 as i64,
            0x3FFF3FFF3FFF3FFFu64 as i64,
        );
        let ct5 = _mm256_set_epi64x(
            0xE000E000E000E000u64 as i64,
            0xE000E000E000E000u64 as i64,
            0xE000E000E000E000u64 as i64,
            0xE000E000E000E000u64 as i64,
        );
        let ct6 = _mm256_set_epi64x(
            0x1FFF1FFF1FFF1FFFu64 as i64,
            0x1FFF1FFF1FFF1FFFu64 as i64,
            0x1FFF1FFF1FFF1FFFu64 as i64,
            0x1FFF1FFF1FFF1FFFu64 as i64,
        );
        let l1 = _mm256_andnot_si256(*el, ct1);
        let l2 = _mm256_srli_epi64(l1, 15);
        let l3 = _mm256_andnot_si256(*el, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        let limb1 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct3);
        let l2 = _mm256_srli_epi64(l1, 14);
        let l3 = _mm256_and_si256(*el, ct4);
        let l4 = _mm256_slli_epi64(l3, 2);
        let limb2 = _mm256_or_si256(l2, l4);
        let l1 = _mm256_and_si256(*el, ct5);
        let l2 = _mm256_srli_epi64(l1, 13);
        let l3 = _mm256_and_si256(*el, ct6);
        let l4 = _mm256_slli_epi64(l3, 3);
        let limb3 = _mm256_or_si256(l2, l4);
        let tmp = _mm256_and_si256(limb1, limb2);
        let tmp = _mm256_and_si256(tmp, limb3);
        let tmp = _mm256_xor_si256(*el, tmp);
        let l1 = _mm256_and_si256(tmp, ct1);
        let l2 = _mm256_srli_epi64(l1, 15);
        let l3 = _mm256_and_si256(tmp, ct2);
        let l4 = _mm256_slli_epi64(l3, 1);
        *el = _mm256_or_si256(l2, l4);
    }
}

// The result is put in (sumhi, sumlo)
#[inline]
unsafe fn add_u128(sumhi: &mut __m256i, sumlo: &mut __m256i, hi: &__m256i, lo: &__m256i) {
    let (r, c) = add64_no_carry(sumlo, lo);
    (*sumhi, _) = add64(sumhi, hi, &c);
    *sumlo = r;
}

/*
// Alternative to add_u128() using non-AVX ops.
unsafe fn add_u128(sumhi: &mut __m256i, sumlo: &mut __m256i, hi: &__m256i, lo: &__m256i) {
    let mut vsh = [0u64; 4];
    let mut vh = [0u64; 4];
    let mut vsl = [0u64; 4];
    let mut vl = [0u64; 4];
    _mm256_storeu_si256(vsh.as_mut_ptr().cast::<__m256i>(), *sumhi);
    _mm256_storeu_si256(vsl.as_mut_ptr().cast::<__m256i>(), *sumlo);
    _mm256_storeu_si256(vh.as_mut_ptr().cast::<__m256i>(), *hi);
    _mm256_storeu_si256(vl.as_mut_ptr().cast::<__m256i>(), *lo);

    for i in 0..4 {
        let r = vsl[i] as u128 + vl[i] as u128 + ((vsh[i] as u128) << 64) + ((vh[i] as u128) << 64);
        vl[i] = r as u64;
        vh[i] = (r >> 64) as u64;
    }
    *sumhi = _mm256_loadu_si256(vh.as_ptr().cast::<__m256i>());
    *sumlo = _mm256_loadu_si256(vl.as_ptr().cast::<__m256i>());
}
*/

// Multiply one u128 (ah, al) by one u64 (b) -> return on u128 (h, l).
// b is small (< 2^32)
// ah is only 0 or 1
// al = alh * 2^32 + all => al * b = alh * b * 2^32 + all * b
// result l = l1 + l2 where l1 = all * b (which is always < 2^64)
// l2 is the low part (<2^64) of alh * b * 2^32 which is (alh * b) << 32
// l1 + l2 may overflow, so we need to keep the carry out
// h = h1 + h2 + h3 where h1 = b if ah is 1 or 0 otherwise
// h2 is the high part of (>2^64) alh * b * 2^32 which is (alh * b) >> 32 which is < 2^32
// h3 is the carry (0 or 1) -- given h1, h2, h3 are < 2^32, h cannot overflow
#[inline]
unsafe fn mul_u128_x_u64(ah: &__m256i, al: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let m = _mm256_cmpeq_epi64(*ah, ones);
    let h = _mm256_and_si256(m, *b);
    let al_h = _mm256_srli_epi64(*al, 32);
    let r_h = _mm256_mul_epu32(al_h, *b);
    let r_l = _mm256_mul_epu32(*al, *b);
    let r_h_l = _mm256_slli_epi64(r_h, 32);
    let (l, carry) = add64_no_carry(&r_l, &r_h_l);
    let r_h_h = _mm256_srli_epi64(r_h, 32);
    let h = _mm256_add_epi64(h, r_h_h);
    let h = _mm256_add_epi64(h, carry);
    (h, l)
}

/*
// Alternative to using non-AVX ops.
unsafe fn mul_u128_x_u64(ah: &__m256i, al: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    let mut val = [0u64; 4];
    let mut vah = [0u64; 4];
    let mut vb = [0u64; 4];
    let mut vh = [0u64; 4];
    let mut vl = [0u64; 4];
    _mm256_storeu_si256(val.as_mut_ptr().cast::<__m256i>(), *al);
    _mm256_storeu_si256(vah.as_mut_ptr().cast::<__m256i>(), *ah);
    _mm256_storeu_si256(vb.as_mut_ptr().cast::<__m256i>(), *b);
    for i in 0..4 {
        let r = (val[i] as u128 + ((vah[i] as u128) << 64)) * (vb[i] as u128);
        vl[i] = r as u64;
        vh[i] = (r >> 64) as u64;
    }
    let h = _mm256_loadu_si256(vh.as_ptr().cast::<__m256i>());
    let l = _mm256_loadu_si256(vl.as_ptr().cast::<__m256i>());
    (h, l)
}
*/

/*
// Alternative to reduce_avx_128_64() using non-AVX ops.
unsafe fn reduce(h: &__m256i, l: &__m256i, s: &mut __m256i) {
    let mut vh = [0u64; 4];
    let mut vl = [0u64; 4];
    let mut v = [0u64; 4];
    _mm256_storeu_si256(vh.as_mut_ptr().cast::<__m256i>(), *h);
    _mm256_storeu_si256(vl.as_mut_ptr().cast::<__m256i>(), *l);

    for i in 0..4 {
        v[i] = GoldilocksField::from_noncanonical_u96((vl[i], vh[i] as u32)).to_noncanonical_u64();
    }

    *s = _mm256_loadu_si256(v.as_ptr().cast::<__m256i>());
}
*/

// (h0, h1, h2) is the high part (>2^64) of the input and can only be 0 or 1 (not higher).
// (l0, l1, l2) is the low part of the input (<2^64).
#[inline]
unsafe fn concrete_avx(
    h0: &__m256i,
    h1: &__m256i,
    h2: &__m256i,
    l0: &mut __m256i,
    l1: &mut __m256i,
    l2: &mut __m256i,
    round_constants: &[u64; SPONGE_WIDTH],
) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let mut sh0 = zeros;
    let mut sh1 = zeros;
    let mut sh2 = zeros;
    let mut sl0 = zeros;
    let mut sl1 = zeros;
    let mut sl2 = zeros;
    for column in 0..SPONGE_WIDTH {
        let mm0 = _mm256_set_epi64x(
            MONOLITH_MAT_12[3][column] as i64,
            MONOLITH_MAT_12[2][column] as i64,
            MONOLITH_MAT_12[1][column] as i64,
            MONOLITH_MAT_12[0][column] as i64,
        );
        let mm1 = _mm256_set_epi64x(
            MONOLITH_MAT_12[7][column] as i64,
            MONOLITH_MAT_12[6][column] as i64,
            MONOLITH_MAT_12[5][column] as i64,
            MONOLITH_MAT_12[4][column] as i64,
        );
        let mm2 = _mm256_set_epi64x(
            MONOLITH_MAT_12[11][column] as i64,
            MONOLITH_MAT_12[10][column] as i64,
            MONOLITH_MAT_12[9][column] as i64,
            MONOLITH_MAT_12[8][column] as i64,
        );
        let tl = match column {
            0 => _mm256_permute4x64_epi64(*l0, 0x0),
            1 => _mm256_permute4x64_epi64(*l0, 0x55),
            2 => _mm256_permute4x64_epi64(*l0, 0xAA),
            3 => _mm256_permute4x64_epi64(*l0, 0xFF),
            4 => _mm256_permute4x64_epi64(*l1, 0x0),
            5 => _mm256_permute4x64_epi64(*l1, 0x55),
            6 => _mm256_permute4x64_epi64(*l1, 0xAA),
            7 => _mm256_permute4x64_epi64(*l1, 0xFF),
            8 => _mm256_permute4x64_epi64(*l2, 0x0),
            9 => _mm256_permute4x64_epi64(*l2, 0x55),
            10 => _mm256_permute4x64_epi64(*l2, 0xAA),
            11 => _mm256_permute4x64_epi64(*l2, 0xFF),
            _ => zeros,
        };
        let th = match column {
            0 => _mm256_permute4x64_epi64(*h0, 0x0),
            1 => _mm256_permute4x64_epi64(*h0, 0x55),
            2 => _mm256_permute4x64_epi64(*h0, 0xAA),
            3 => _mm256_permute4x64_epi64(*h0, 0xFF),
            4 => _mm256_permute4x64_epi64(*h1, 0x0),
            5 => _mm256_permute4x64_epi64(*h1, 0x55),
            6 => _mm256_permute4x64_epi64(*h1, 0xAA),
            7 => _mm256_permute4x64_epi64(*h1, 0xFF),
            8 => _mm256_permute4x64_epi64(*h2, 0x0),
            9 => _mm256_permute4x64_epi64(*h2, 0x55),
            10 => _mm256_permute4x64_epi64(*h2, 0xAA),
            11 => _mm256_permute4x64_epi64(*h2, 0xFF),
            _ => zeros,
        };

        // change to simple mul
        let (mh0, ml0) = mul_u128_x_u64(&th, &tl, &mm0);
        let (mh1, ml1) = mul_u128_x_u64(&th, &tl, &mm1);
        let (mh2, ml2) = mul_u128_x_u64(&th, &tl, &mm2);

        // add with carry
        add_u128(&mut sh0, &mut sl0, &mh0, &ml0);
        add_u128(&mut sh1, &mut sl1, &mh1, &ml1);
        add_u128(&mut sh2, &mut sl2, &mh2, &ml2);
    }

    // add round constants
    let rc0 = _mm256_loadu_si256(round_constants[0..4].as_ptr().cast::<__m256i>());
    let rc1 = _mm256_loadu_si256(round_constants[4..8].as_ptr().cast::<__m256i>());
    let rc2 = _mm256_loadu_si256(round_constants[8..12].as_ptr().cast::<__m256i>());
    add_u128(&mut sh0, &mut sl0, &zeros, &rc0);
    add_u128(&mut sh1, &mut sl1, &zeros, &rc1);
    add_u128(&mut sh2, &mut sl2, &zeros, &rc2);

    // reduce u128 to u64 Goldilocks
    *l0 = reduce_avx_96_64(&sh0, &sl0);
    *l1 = reduce_avx_96_64(&sh1, &sl1);
    *l2 = reduce_avx_96_64(&sh2, &sl2);
}

// The high part (h0, h1, h2) is only for output (there is no high part from the previous operation).
// Note that h can only be 0 or 1, not higher.
unsafe fn bricks_avx(
    h0: &mut __m256i,
    h1: &mut __m256i,
    h2: &mut __m256i,
    l0: &mut __m256i,
    l1: &mut __m256i,
    l2: &mut __m256i,
) {
    // get prev using permute and blend
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let ss0 = _mm256_permute4x64_epi64(*l0, 0x90);
    let ss1 = _mm256_permute4x64_epi64(*l1, 0x93);
    let ss2 = _mm256_permute4x64_epi64(*l2, 0x93);
    let ss3 = _mm256_permute4x64_epi64(*l0, 0x3);
    let ss4 = _mm256_permute4x64_epi64(*l1, 0x3);
    let ss0 = _mm256_blend_epi32(ss0, zeros, 0x3);
    let ss1 = _mm256_blend_epi32(ss1, ss3, 0x3);
    let ss2 = _mm256_blend_epi32(ss2, ss4, 0x3);

    // square
    let p0 = sqr_avx(&ss0);
    let p1 = sqr_avx(&ss1);
    let p2 = sqr_avx(&ss2);

    // add
    (*l0, *h0) = add64_no_carry(l0, &p0);
    (*l1, *h1) = add64_no_carry(l1, &p1);
    (*l2, *h2) = add64_no_carry(l2, &p2);
}

#[inline(always)]
unsafe fn block1_avx(x: &__m256i, y: [i64; 3]) -> __m256i {
    let x0 = _mm256_permute4x64_epi64(*x, 0x0);
    let x1 = _mm256_permute4x64_epi64(*x, 0x55);
    let x2 = _mm256_permute4x64_epi64(*x, 0xAA);

    let f0 = _mm256_set_epi64x(0, y[2], y[1], y[0]);
    let f1 = _mm256_set_epi64x(0, y[1], y[0], y[2]);
    let f2 = _mm256_set_epi64x(0, y[0], y[2], y[1]);

    let t0 = mul64_no_overflow(&x0, &f0);
    let t1 = mul64_no_overflow(&x1, &f1);
    let t2 = mul64_no_overflow(&x2, &f2);

    let t0 = _mm256_add_epi64(t0, t1);
    _mm256_add_epi64(t0, t2)
}

#[allow(dead_code)]
#[inline(always)]
unsafe fn block2_full_avx(xr: &__m256i, xi: &__m256i, y: [(i64, i64); 3]) -> (__m256i, __m256i) {
    let yr = _mm256_set_epi64x(0, y[2].0, y[1].0, y[0].0);
    let yi = _mm256_set_epi64x(0, y[2].1, y[1].1, y[0].1);
    let ys = _mm256_add_epi64(yr, yi);
    let xs = _mm256_add_epi64(*xr, *xi);

    // z0
    // z0r = dif2[0] + prod[1] - sum[1] + prod[2] - sum[2]
    // z0i = prod[0] - sum[0] + dif1[1] + dif1[2]
    let yy = _mm256_permute4x64_epi64(yr, 0x18);
    let mr_z0 = mul64_no_overflow(xr, &yy);
    let yy = _mm256_permute4x64_epi64(yi, 0x18);
    let mi_z0 = mul64_no_overflow(xi, &yy);
    let sum = _mm256_add_epi64(mr_z0, mi_z0);
    let dif1 = _mm256_sub_epi64(mi_z0, mr_z0);
    let dif2 = _mm256_sub_epi64(mr_z0, mi_z0);
    let yy = _mm256_permute4x64_epi64(ys, 0x18);
    let prod = mul64_no_overflow(&xs, &yy);
    let dif3 = _mm256_sub_epi64(prod, sum);
    let dif3perm1 = _mm256_permute4x64_epi64(dif3, 0x1);
    let dif3perm2 = _mm256_permute4x64_epi64(dif3, 0x2);
    let z0r = _mm256_add_epi64(dif2, dif3perm1);
    let z0r = _mm256_add_epi64(z0r, dif3perm2);
    let dif1perm1 = _mm256_permute4x64_epi64(dif1, 0x1);
    let dif1perm2 = _mm256_permute4x64_epi64(dif1, 0x2);
    let z0i = _mm256_add_epi64(dif3, dif1perm1);
    let z0i = _mm256_add_epi64(z0i, dif1perm2);
    let mask  = _mm256_set_epi64x(0, 0, 0, 0xFFFFFFFFFFFFFFFFu64 as i64);
    let z0r = _mm256_and_si256(z0r, mask);
    let z0i = _mm256_and_si256(z0i, mask);

    // z1
    // z1r = dif2[0] + dif2[1] + prod[2] - sum[2];
    // z1i = prod[0] - sum[0] + prod[1] - sum[1] + dif1[2];
    let yy = _mm256_permute4x64_epi64(yr, 0x21);
    let mr_z1 = mul64_no_overflow(xr, &yy);
    let yy = _mm256_permute4x64_epi64(yi, 0x21);
    let mi_z1 = mul64_no_overflow(xi, &yy);
    let sum = _mm256_add_epi64(mr_z1, mi_z1);
    let dif1 = _mm256_sub_epi64(mi_z1, mr_z1);
    let dif2 = _mm256_sub_epi64(mr_z1, mi_z1);
    let yy = _mm256_permute4x64_epi64(ys, 0x21);
    let prod = mul64_no_overflow(&xs, &yy);
    let dif3 = _mm256_sub_epi64(prod, sum);
    let dif2perm = _mm256_permute4x64_epi64(dif2, 0x0);
    let dif3perm = _mm256_permute4x64_epi64(dif3, 0x8);
    let z1r = _mm256_add_epi64(dif2, dif2perm);
    let z1r = _mm256_add_epi64(z1r, dif3perm);
    let dif3perm = _mm256_permute4x64_epi64(dif3, 0x0);
    let dif1perm = _mm256_permute4x64_epi64(dif1, 0x8);
    let z1i = _mm256_add_epi64(dif3, dif3perm);
    let z1i = _mm256_add_epi64(z1i, dif1perm);
    let mask  = _mm256_set_epi64x(0, 0, 0xFFFFFFFFFFFFFFFFu64 as i64, 0);
    let z1r = _mm256_and_si256(z1r, mask);
    let z1i = _mm256_and_si256(z1i, mask);

    // z2
    // z2r = dif2[0] + dif2[1] + dif2[2];
    // z2i = prod[0] - sum[0] + prod[1] - sum[1] + prod[2] - sum[2]
    let yy = _mm256_permute4x64_epi64(yr, 0x6);
    let mr_z2 = mul64_no_overflow(xr, &yy);
    let yy = _mm256_permute4x64_epi64(yi, 0x6);
    let mi_z2 = mul64_no_overflow(xi, &yy);
    let sum = _mm256_add_epi64(mr_z2, mi_z2);
    let dif2 = _mm256_sub_epi64(mr_z2, mi_z2);
    let yy = _mm256_permute4x64_epi64(ys, 0x6);
    let prod = mul64_no_overflow(&xs, &yy);
    let dif3 = _mm256_sub_epi64(prod, sum);
    let dif2perm1 = _mm256_permute4x64_epi64(dif2, 0x0);
    let dif2perm2 = _mm256_permute4x64_epi64(dif2, 0x10);
    let z2r = _mm256_add_epi64(dif2, dif2perm1);
    let z2r = _mm256_add_epi64(z2r, dif2perm2);
    let dif3perm1 = _mm256_permute4x64_epi64(dif3, 0x0);
    let dif3perm2 = _mm256_permute4x64_epi64(dif3, 0x10);
    let z2i = _mm256_add_epi64(dif3, dif3perm1);
    let z2i = _mm256_add_epi64(z2i, dif3perm2);
    let mask  = _mm256_set_epi64x(0, 0xFFFFFFFFFFFFFFFFu64 as i64, 0, 0);
    let z2r = _mm256_and_si256(z2r, mask);
    let z2i = _mm256_and_si256(z2i, mask);

    let zr = _mm256_or_si256(z0r, z1r);
    let zr = _mm256_or_si256(zr, z2r);
    let zi = _mm256_or_si256(z0i, z1i);
    let zi = _mm256_or_si256(zi, z2i);
    (zr, zi)
}

#[inline(always)]
unsafe fn block2_avx(xr: &__m256i, xi: &__m256i, y: [(i64, i64); 3]) -> (__m256i, __m256i) {
    let mut vxr: [i64; 4] = [0; 4];
    let mut vxi: [i64; 4] = [0; 4];
    _mm256_storeu_si256(vxr.as_mut_ptr().cast::<__m256i>(), *xr);
    _mm256_storeu_si256(vxi.as_mut_ptr().cast::<__m256i>(), *xi);
    let x: [(i64, i64); 3] = [(vxr[0], vxi[0]), (vxr[1], vxi[1]), (vxr[2], vxi[2])];
    let b = block2(x, y);
    vxr = [b[0].0, b[1].0, b[2].0, 0];
    vxi = [b[0].1, b[1].1, b[2].1, 0];
    let rr = _mm256_loadu_si256(vxr.as_ptr().cast::<__m256i>());
    let ri = _mm256_loadu_si256(vxi.as_ptr().cast::<__m256i>());
    (rr, ri)
}

#[inline(always)]
unsafe fn block3_avx(x: &__m256i, y: [i64; 3]) -> __m256i {
    let x0 = _mm256_permute4x64_epi64(*x, 0x0);
    let x1 = _mm256_permute4x64_epi64(*x, 0x55);
    let x2 = _mm256_permute4x64_epi64(*x, 0xAA);

    let f0 = _mm256_set_epi64x(0, y[2], y[1], y[0]);
    let f1 = _mm256_set_epi64x(0, y[1], y[0], -y[2]);
    let f2 = _mm256_set_epi64x(0, y[0], -y[2], -y[1]);

    let t0 = mul64_no_overflow(&x0, &f0);
    let t1 = mul64_no_overflow(&x1, &f1);
    let t2 = mul64_no_overflow(&x2, &f2);

    let t0 = _mm256_add_epi64(t0, t1);
    _mm256_add_epi64(t0, t2)
}

#[inline(always)]
unsafe fn fft2_real_avx(x0: &__m256i, x1: &__m256i) -> (__m256i, __m256i) {
    let y0 = _mm256_add_epi64(*x0, *x1);
    let y1 = _mm256_sub_epi64(*x0, *x1);
    (y0, y1)
}

#[inline(always)]
unsafe fn fft4_real_avx(
    x0: &__m256i,
    x1: &__m256i,
    x2: &__m256i,
    x3: &__m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let (z0, z2) = fft2_real_avx(x0, x2);
    let (z1, z3) = fft2_real_avx(x1, x3);
    let y0 = _mm256_add_epi64(z0, z1);
    let y2 = _mm256_sub_epi64(z0, z1);
    let y3 = _mm256_sub_epi64(zeros, z3);
    (y0, z2, y3, y2)
}

#[inline(always)]
unsafe fn ifft2_real_unreduced_avx(y0: &__m256i, y1: &__m256i) -> (__m256i, __m256i) {
    let x0 = _mm256_add_epi64(*y0, *y1);
    let x1 = _mm256_sub_epi64(*y0, *y1);
    (x0, x1)
}

#[inline(always)]
unsafe fn ifft4_real_unreduced_avx(
    y: (__m256i, (__m256i, __m256i), __m256i),
) -> (__m256i, __m256i, __m256i, __m256i) {
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let z0 = _mm256_add_epi64(y.0, y.2);
    let z1 = _mm256_sub_epi64(y.0, y.2);
    let z2 = y.1 .0;
    let z3 = _mm256_sub_epi64(zeros, y.1 .1);
    let (x0, x2) = ifft2_real_unreduced_avx(&z0, &z2);
    let (x1, x3) = ifft2_real_unreduced_avx(&z1, &z3);
    (x0, x1, x2, x3)
}

#[inline]
unsafe fn mds_multiply_freq_avx(s0: &mut __m256i, s1: &mut __m256i, s2: &mut __m256i) {
    /*
    // Alternative code using store and set.
    let mut s: [i64; 12] = [0; 12];
    _mm256_storeu_si256(s[0..4].as_mut_ptr().cast::<__m256i>(), *s0);
    _mm256_storeu_si256(s[4..8].as_mut_ptr().cast::<__m256i>(), *s1);
    _mm256_storeu_si256(s[8..12].as_mut_ptr().cast::<__m256i>(), *s2);
    let f0 = _mm256_set_epi64x(0, s[2], s[1], s[0]);
    let f1 = _mm256_set_epi64x(0, s[5], s[4], s[3]);
    let f2 = _mm256_set_epi64x(0, s[8], s[7], s[6]);
    let f3 = _mm256_set_epi64x(0, s[11], s[10], s[9]);
    */

    // Alternative code using permute and blend (it is faster).
    let f0 = *s0;
    let f11 = _mm256_permute4x64_epi64(*s0, 0x3);
    let f12 = _mm256_permute4x64_epi64(*s1, 0x10);
    let f1 = _mm256_blend_epi32(f11, f12, 0x3C);
    let f21 = _mm256_permute4x64_epi64(*s1, 0xE);
    let f22 = _mm256_permute4x64_epi64(*s2, 0x0);
    let f2 = _mm256_blend_epi32(f21, f22, 0x30);
    let f3 = _mm256_permute4x64_epi64(*s2, 0x39);

    let (u0, u1, u2, u3) = fft4_real_avx(&f0, &f1, &f2, &f3);

    // let [v0, v4, v8] = block1_avx([u[0], u[1], u[2]], MDS_FREQ_BLOCK_ONE);
    // [u[0], u[1], u[2]] are all in u0
    let f0 = block1_avx(&u0, MDS_FREQ_BLOCK_ONE);

    // let [v1, v5, v9] = block2([(u[0], v[0]), (u[1], v[1]), (u[2], v[2])], MDS_FREQ_BLOCK_TWO);
    let (f1, f2) = block2_avx(&u1, &u2, MDS_FREQ_BLOCK_TWO);

    // let [v2, v6, v10] = block3_avx([u[0], u[1], u[2]], MDS_FREQ_BLOCK_ONE);
    // [u[0], u[1], u[2]] are all in u3
    let f3 = block3_avx(&u3, MDS_FREQ_BLOCK_THREE);

    let (r0, r3, r6, r9) = ifft4_real_unreduced_avx((f0, (f1, f2), f3));
    let t = _mm256_permute4x64_epi64(r3, 0x0);
    *s0 = _mm256_blend_epi32(r0, t, 0xC0);
    let t1 = _mm256_permute4x64_epi64(r3, 0x9);
    let t2 = _mm256_permute4x64_epi64(r6, 0x40);
    *s1 = _mm256_blend_epi32(t1, t2, 0xF0);
    let t1 = _mm256_permute4x64_epi64(r6, 0x2);
    let t2 = _mm256_permute4x64_epi64(r9, 0x90);
    *s2 = _mm256_blend_epi32(t1, t2, 0xFC);
}

#[inline]
unsafe fn concrete_avx_freq(
    h0: &__m256i,
    h1: &__m256i,
    h2: &__m256i,
    l0: &mut __m256i,
    l1: &mut __m256i,
    l2: &mut __m256i,
    round_constants: &[u64; SPONGE_WIDTH],
) {
    let mask32 = _mm256_set_epi64x(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
    let mut sh0 = _mm256_srli_epi64(*l0, 32);
    let mut sh1 = _mm256_srli_epi64(*l1, 32);
    let mut sh2 = _mm256_srli_epi64(*l2, 32);
    let shh0 = _mm256_slli_epi64(*h0, 32);
    let shh1 = _mm256_slli_epi64(*h1, 32);
    let shh2 = _mm256_slli_epi64(*h2, 32);
    sh0 = _mm256_or_si256(sh0, shh0);
    sh1 = _mm256_or_si256(sh1, shh1);
    sh2 = _mm256_or_si256(sh2, shh2);
    *l0 = _mm256_and_si256(*l0, mask32);
    *l1 = _mm256_and_si256(*l1, mask32);
    *l2 = _mm256_and_si256(*l2, mask32);

    mds_multiply_freq_avx(&mut sh0, &mut sh1, &mut sh2);
    mds_multiply_freq_avx(l0, l1, l2);

    let mut sl0 = _mm256_slli_epi64(sh0, 32);
    let mut sl1 = _mm256_slli_epi64(sh1, 32);
    let mut sl2 = _mm256_slli_epi64(sh2, 32);
    let shl0 = _mm256_srli_epi64(sh0, 32);
    let shl1 = _mm256_srli_epi64(sh1, 32);
    let shl2 = _mm256_srli_epi64(sh2, 32);

    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    sh0 = zeros;
    sh1 = zeros;
    sh2 = zeros;
    add_u128(&mut sh0, &mut sl0, &shl0, l0);
    add_u128(&mut sh1, &mut sl1, &shl1, l1);
    add_u128(&mut sh2, &mut sl2, &shl2, l2);
    let rc0 = _mm256_loadu_si256(round_constants[0..4].as_ptr().cast::<__m256i>());
    let rc1 = _mm256_loadu_si256(round_constants[4..8].as_ptr().cast::<__m256i>());
    let rc2 = _mm256_loadu_si256(round_constants[8..12].as_ptr().cast::<__m256i>());
    add_u128(&mut sh0, &mut sl0, &zeros, &rc0);
    add_u128(&mut sh1, &mut sl1, &zeros, &rc1);
    add_u128(&mut sh2, &mut sl2, &zeros, &rc2);

    // reduce u128 to u64 Goldilocks
    *l0 = reduce_avx_96_64(&sh0, &sl0);
    *l1 = reduce_avx_96_64(&sh1, &sl1);
    *l2 = reduce_avx_96_64(&sh2, &sl2);
}

// input is obtained via to_noncanonical_u64()
#[inline]
pub fn monolith_avx(state: &mut [u64; SPONGE_WIDTH]) {
    unsafe {
        let zeros = _mm256_set_epi64x(0, 0, 0, 0);

        // low part of the state (< 2^64)
        let mut sl0 = _mm256_loadu_si256(state[0..4].as_ptr().cast::<__m256i>());
        let mut sl1 = _mm256_loadu_si256(state[4..8].as_ptr().cast::<__m256i>());
        let mut sl2 = _mm256_loadu_si256(state[8..12].as_ptr().cast::<__m256i>());

        // high part of the state (only after bricks operations)
        let mut sh0 = zeros;
        let mut sh1 = zeros;
        let mut sh2 = zeros;

        // rounds
        concrete_avx(
            &sh0,
            &sh1,
            &sh2,
            &mut sl0,
            &mut sl1,
            &mut sl2,
            &MONOLITH_ROUND_CONSTANTS[0],
        );
        for rc in MONOLITH_ROUND_CONSTANTS.iter().skip(1) {
            bar_avx(&mut sl0);
            bricks_avx(&mut sh0, &mut sh1, &mut sh2, &mut sl0, &mut sl1, &mut sl2);

            #[cfg(not(feature = "default-sponge-params"))]
            concrete_avx(&sh0, &sh1, &sh2, &mut sl0, &mut sl1, &mut sl2, rc);

            #[cfg(feature = "default-sponge-params")]
            concrete_avx_freq(&sh0, &sh1, &sh2, &mut sl0, &mut sl1, &mut sl2, rc);
        }

        // store states
        _mm256_storeu_si256(state[0..4].as_mut_ptr().cast::<__m256i>(), sl0);
        _mm256_storeu_si256(state[4..8].as_mut_ptr().cast::<__m256i>(), sl1);
        _mm256_storeu_si256(state[8..12].as_mut_ptr().cast::<__m256i>(), sl2);
    }
}
