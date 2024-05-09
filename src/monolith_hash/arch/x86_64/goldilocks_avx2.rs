use core::arch::x86_64::*;

pub(crate) const MSB_1: i64 = 0x8000000000000000u64 as i64;
pub(crate) const P_N_1: i64 = 0xFFFFFFFF;

#[inline(always)]
pub fn add_avx_s_b_small(a_s: &__m256i, b_small: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_add_epi64(*a_s, *b_small);
        let mask_ = _mm256_cmpgt_epi32(*a_s, c0_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_add_epi64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn sub_avx_s_b_small(a_s: &__m256i, b: &__m256i) -> __m256i {
    unsafe {
        let c0_s = _mm256_sub_epi64(*a_s, *b);
        let mask_ = _mm256_cmpgt_epi32(c0_s, *a_s);
        let corr_ = _mm256_srli_epi64(mask_, 32);
        _mm256_sub_epi64(c0_s, corr_)
    }
}

#[inline(always)]
pub fn reduce_avx_128_64(c_h: &__m256i, c_l: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        let c_hh = _mm256_srli_epi64(*c_h, 32);
        let c_ls = _mm256_xor_si256(*c_l, msb);
        let c1_s = sub_avx_s_b_small(&c_ls, &c_hh);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let c2 = _mm256_mul_epu32(*c_h, p_n);
        let c_s = add_avx_s_b_small(&c1_s, &c2);
        _mm256_xor_si256(c_s, msb)
    }
}

// Here we suppose c_h < 2^32
#[inline(always)]
pub fn reduce_avx_96_64(c_h: &__m256i, c_l: &__m256i) -> __m256i {
    unsafe {
        let msb = _mm256_set_epi64x(MSB_1, MSB_1, MSB_1, MSB_1);
        let p_n = _mm256_set_epi64x(P_N_1, P_N_1, P_N_1, P_N_1);
        let c_ls = _mm256_xor_si256(*c_l, msb);
        let c2 = _mm256_mul_epu32(*c_h, p_n);
        let c_s = add_avx_s_b_small(&c_ls, &c2);
        _mm256_xor_si256(c_s, msb)
    }
}

#[inline(always)]
pub fn sqr_avx_128(a: &__m256i) -> (__m256i, __m256i) {
    unsafe {
        let a_h = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(*a)));
        let c_ll = _mm256_mul_epu32(*a, *a);
        let c_lh = _mm256_mul_epu32(*a, a_h);
        let c_hh = _mm256_mul_epu32(a_h, a_h);
        let c_ll_hi = _mm256_srli_epi64(c_ll, 33);
        let t0 = _mm256_add_epi64(c_lh, c_ll_hi);
        let t0_hi = _mm256_srli_epi64(t0, 31);
        let res_hi = _mm256_add_epi64(c_hh, t0_hi);
        let c_lh_lo = _mm256_slli_epi64(c_lh, 33);
        let res_lo = _mm256_add_epi64(c_ll, c_lh_lo);
        (res_hi, res_lo)
    }
}

#[inline(always)]
pub fn sqr_avx(a: &__m256i) -> __m256i {
    let (c_h, c_l) = sqr_avx_128(a);
    reduce_avx_128_64(&c_h, &c_l)
}
