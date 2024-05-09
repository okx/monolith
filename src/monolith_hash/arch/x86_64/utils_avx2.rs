use core::arch::x86_64::*;

#[inline]
pub unsafe fn add64_no_carry(a: &__m256i, b: &__m256i) -> (__m256i, __m256i) {
    /*
     * a and b are signed 4 x i64. Suppose a and b represent only one i64, then:
     * - (test 1): if a < 2^63 and b < 2^63 (this means a >= 0 and b >= 0) => sum does not overflow => cout = 0
     * - if a >= 2^63 and b >= 2^63 => sum overflows so sum = a + b and cout = 1
     * - (test 2): if (a < 2^63 and b >= 2^63) or (a >= 2^63 and b < 2^63)
     *   - (test 3): if a + b < 2^64 (this means a + b is negative in signed representation) => no overflow so cout = 0
     *   - (test 3): if a + b >= 2^64 (this means a + b becomes positive in signed representation, that is, a + b >= 0) => there is overflow so cout = 1
     */
    let ones = _mm256_set_epi64x(1, 1, 1, 1);
    let zeros = _mm256_set_epi64x(0, 0, 0, 0);
    let r = _mm256_add_epi64(*a, *b);
    let ma = _mm256_cmpgt_epi64(zeros, *a);
    let mb = _mm256_cmpgt_epi64(zeros, *b);
    let m1 = _mm256_and_si256(ma, mb); // test 1
    let m21 = _mm256_andnot_si256(ma, mb);
    let m22 = _mm256_andnot_si256(mb, ma);
    let m2 = _mm256_or_si256(m21, m22); // test 2
    let m23 = _mm256_cmpgt_epi64(zeros, r); // test 3
    let m2 = _mm256_andnot_si256(m23, m2);
    let m = _mm256_or_si256(m1, m2);
    let co = _mm256_and_si256(m, ones);
    (r, co)
}

// cin is carry in and must be 0 or 1
#[inline]
pub unsafe fn add64(a: &__m256i, b: &__m256i, cin: &__m256i) -> (__m256i, __m256i) {
    let (r1, c1) = add64_no_carry(a, b);
    let max = _mm256_set_epi64x(-1, -1, -1, -1);
    let m = _mm256_cmpeq_epi64(r1, max);
    let r = _mm256_add_epi64(r1, *cin);
    let m = _mm256_and_si256(*cin, m);
    let co = _mm256_or_si256(m, c1);
    (r, co)
}

// Multiply two 64bit numbers with the assumption that the product does not averflow.
#[inline]
pub unsafe fn mul64_no_overflow(a: &__m256i, b: &__m256i) -> __m256i {
    let r = _mm256_mul_epu32(*a, *b);
    let ah = _mm256_srli_epi64(*a, 32);
    let bh = _mm256_srli_epi64(*b, 32);
    let r1 = _mm256_mul_epu32(*a, bh);
    let r1 = _mm256_slli_epi64(r1, 32);
    let r = _mm256_add_epi64(r, r1);
    let r1 = _mm256_mul_epu32(ah, *b);
    let r1 = _mm256_slli_epi64(r1, 32);
    let r = _mm256_add_epi64(r, r1);
    r
}