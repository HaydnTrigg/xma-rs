// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavutil/x86/tx_float.asm
// and libavutil/x86/x86util.asm (FFmpeg n9.1-dev-56-gae4314e2f4); the original files carry these
// notices:
//
// Copyright (c) Lynne
// Copyright (C) 2008-2010 x264 project
// Copyright (c) 2026 the xma authors (the Rust port)
//
// xma is free software; you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation; either version
// 2.1 of the License, or (at your option) any later version.
//
// xma is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License (the LICENSE file) for more details.

//! The instructions FFmpeg's AVX2 transforms are written in, as one interface with two
//! implementations: the instructions themselves ([`Avx2`]), and each instruction's exact
//! arithmetic on plain arrays ([`Portable`]).
//!
//! The two give the same bits for the same input. Every lane operation is a single IEEE
//! operation in round-to-nearest (an add, a multiply, or a fused multiply-add rounded once,
//! which `f32::mul_add` is), and every shuffle moves bits. So the transforms written against
//! [`Simd`] compute what FFmpeg's assembly computes on any machine, and the AVX2 build is only
//! the fast way of doing it.
//!
//! A vector is 8 floats in two 128-bit lanes; the in-lane shuffles act on each lane alike.

/// `qABCD`, the shuffle immediate of x86util.asm: element 0 from `D`, 1 from `C`, 2 from `B`,
/// 3 from `A`.
pub(crate) const fn q(a: i32, b: i32, c: i32, d: i32) -> i32 {
    (a << 6) | (b << 4) | (c << 2) | d
}

/// The instructions the transforms use. Every method is the instruction of its name; the
/// three-operand forms are `dst = op(a, b)`.
pub(crate) trait Simd: Copy {
    type V: Copy;

    /// `movups` from 8 floats.
    unsafe fn load(at: *const f32) -> Self::V;
    /// `movups` to 8 floats.
    unsafe fn store(at: *mut f32, value: Self::V);
    /// `vextractf128 [at], value, 0`.
    unsafe fn store_low(at: *mut f32, value: Self::V);
    /// `vextractf128 [at], value, 1`.
    unsafe fn store_high(at: *mut f32, value: Self::V);
    /// A constant.
    fn constant(values: &[f32; 8]) -> Self::V;
    /// A constant of sign bits (the `mask_*` tables).
    fn mask(bits: &[u32; 8]) -> Self::V;
    fn zero() -> Self::V;

    fn add(a: Self::V, b: Self::V) -> Self::V;
    fn sub(a: Self::V, b: Self::V) -> Self::V;
    fn mul(a: Self::V, b: Self::V) -> Self::V;
    fn xor(a: Self::V, b: Self::V) -> Self::V;
    /// `addsubps`: even elements `a - b`, odd `a + b`.
    fn addsub(a: Self::V, b: Self::V) -> Self::V;
    /// `vfmaddsubps`: even elements `a * b - c`, odd `a * b + c`, rounded once.
    fn fmaddsub(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// `vfmsubaddps`: even elements `a * b + c`, odd `a * b - c`, rounded once.
    fn fmsubadd(a: Self::V, b: Self::V, c: Self::V) -> Self::V;
    /// `vfmaddps`: `a * b + c`, rounded once.
    fn fmadd(a: Self::V, b: Self::V, c: Self::V) -> Self::V;

    /// `shufps`.
    fn shufps<const IMM: i32>(a: Self::V, b: Self::V) -> Self::V;
    /// `unpcklpd`.
    fn unpcklpd(a: Self::V, b: Self::V) -> Self::V;
    /// `unpckhpd`.
    fn unpckhpd(a: Self::V, b: Self::V) -> Self::V;
    /// `movsldup`.
    fn movsldup(a: Self::V) -> Self::V;
    /// `movshdup`.
    fn movshdup(a: Self::V) -> Self::V;
    /// `vpermilps` with a vector of indices.
    fn permilps(a: Self::V, indices: &[i32; 8]) -> Self::V;
    /// `vperm2f128`.
    fn perm2f128<const IMM: i32>(a: Self::V, b: Self::V) -> Self::V;
    /// `vpermpd` (the four 64-bit elements).
    fn permpd<const IMM: i32>(a: Self::V) -> Self::V;
    /// `blendps`.
    fn blendps<const IMM: i32>(a: Self::V, b: Self::V) -> Self::V;
}

/// The instructions' arithmetic on arrays.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Portable;

type A = [f32; 8];

#[inline(always)]
fn each(f: impl Fn(usize) -> f32) -> A {
    std::array::from_fn(f)
}

impl Simd for Portable {
    type V = A;

    #[inline(always)]
    unsafe fn load(at: *const f32) -> A {
        // SAFETY: the caller's 8 floats.
        unsafe { at.cast::<A>().read_unaligned() }
    }
    #[inline(always)]
    unsafe fn store(at: *mut f32, value: A) {
        // SAFETY: the caller's 8 floats.
        unsafe { at.cast::<A>().write_unaligned(value) }
    }
    #[inline(always)]
    unsafe fn store_low(at: *mut f32, value: A) {
        // SAFETY: the caller's 4 floats.
        unsafe {
            at.cast::<[f32; 4]>()
                .write_unaligned([value[0], value[1], value[2], value[3]])
        }
    }
    #[inline(always)]
    unsafe fn store_high(at: *mut f32, value: A) {
        // SAFETY: the caller's 4 floats.
        unsafe {
            at.cast::<[f32; 4]>()
                .write_unaligned([value[4], value[5], value[6], value[7]])
        }
    }
    #[inline(always)]
    fn constant(values: &[f32; 8]) -> A {
        *values
    }
    #[inline(always)]
    fn mask(bits: &[u32; 8]) -> A {
        each(|i| f32::from_bits(bits[i]))
    }
    #[inline(always)]
    fn zero() -> A {
        [0.0; 8]
    }

    #[inline(always)]
    fn add(a: A, b: A) -> A {
        each(|i| a[i] + b[i])
    }
    #[inline(always)]
    fn sub(a: A, b: A) -> A {
        each(|i| a[i] - b[i])
    }
    #[inline(always)]
    fn mul(a: A, b: A) -> A {
        each(|i| a[i] * b[i])
    }
    #[inline(always)]
    fn xor(a: A, b: A) -> A {
        each(|i| f32::from_bits(a[i].to_bits() ^ b[i].to_bits()))
    }
    #[inline(always)]
    fn addsub(a: A, b: A) -> A {
        each(|i| if i % 2 == 0 { a[i] - b[i] } else { a[i] + b[i] })
    }
    #[inline(always)]
    fn fmaddsub(a: A, b: A, c: A) -> A {
        each(|i| {
            if i % 2 == 0 {
                a[i].mul_add(b[i], -c[i])
            } else {
                a[i].mul_add(b[i], c[i])
            }
        })
    }
    #[inline(always)]
    fn fmsubadd(a: A, b: A, c: A) -> A {
        each(|i| {
            if i % 2 == 0 {
                a[i].mul_add(b[i], c[i])
            } else {
                a[i].mul_add(b[i], -c[i])
            }
        })
    }
    #[inline(always)]
    fn fmadd(a: A, b: A, c: A) -> A {
        each(|i| a[i].mul_add(b[i], c[i]))
    }

    #[inline(always)]
    fn shufps<const IMM: i32>(a: A, b: A) -> A {
        each(|i| {
            let lane = i & 4;
            let select = ((IMM >> (2 * (i & 3))) & 3) as usize;
            if i & 3 < 2 {
                a[lane + select]
            } else {
                b[lane + select]
            }
        })
    }
    #[inline(always)]
    fn unpcklpd(a: A, b: A) -> A {
        [a[0], a[1], b[0], b[1], a[4], a[5], b[4], b[5]]
    }
    #[inline(always)]
    fn unpckhpd(a: A, b: A) -> A {
        [a[2], a[3], b[2], b[3], a[6], a[7], b[6], b[7]]
    }
    #[inline(always)]
    fn movsldup(a: A) -> A {
        [a[0], a[0], a[2], a[2], a[4], a[4], a[6], a[6]]
    }
    #[inline(always)]
    fn movshdup(a: A) -> A {
        [a[1], a[1], a[3], a[3], a[5], a[5], a[7], a[7]]
    }
    #[inline(always)]
    fn permilps(a: A, indices: &[i32; 8]) -> A {
        each(|i| a[(i & 4) + (indices[i] & 3) as usize])
    }
    #[inline(always)]
    fn perm2f128<const IMM: i32>(a: A, b: A) -> A {
        let half = |select: i32| -> [f32; 4] {
            let source = if select & 2 == 0 { &a } else { &b };
            let at = if select & 1 == 0 { 0 } else { 4 };
            if select & 8 != 0 {
                [0.0; 4]
            } else {
                [source[at], source[at + 1], source[at + 2], source[at + 3]]
            }
        };
        let low = half(IMM & 0xF);
        let high = half((IMM >> 4) & 0xF);
        [
            low[0], low[1], low[2], low[3], high[0], high[1], high[2], high[3],
        ]
    }
    #[inline(always)]
    fn permpd<const IMM: i32>(a: A) -> A {
        each(|i| {
            let select = ((IMM >> (2 * (i / 2))) & 3) as usize;
            a[select * 2 + (i & 1)]
        })
    }
    #[inline(always)]
    fn blendps<const IMM: i32>(a: A, b: A) -> A {
        each(|i| if (IMM >> i) & 1 != 0 { b[i] } else { a[i] })
    }
}

/// The instructions themselves. Only used inside functions compiled with `avx2` and `fma`
/// enabled ([`super::Kernel::Avx2`] checks the processor for them).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx2;

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    use super::{Avx2, Simd};

    // SAFETY (every function): called only from code compiled with avx2 and fma, on a processor
    // that has them; the pointers are the caller's.
    impl Simd for Avx2 {
        type V = __m256;

        #[inline(always)]
        unsafe fn load(at: *const f32) -> __m256 {
            unsafe { _mm256_loadu_ps(at) }
        }
        #[inline(always)]
        unsafe fn store(at: *mut f32, value: __m256) {
            unsafe { _mm256_storeu_ps(at, value) }
        }
        #[inline(always)]
        unsafe fn store_low(at: *mut f32, value: __m256) {
            unsafe { _mm_storeu_ps(at, _mm256_castps256_ps128(value)) }
        }
        #[inline(always)]
        unsafe fn store_high(at: *mut f32, value: __m256) {
            unsafe { _mm_storeu_ps(at, _mm256_extractf128_ps::<1>(value)) }
        }
        #[inline(always)]
        fn constant(values: &[f32; 8]) -> __m256 {
            unsafe { _mm256_loadu_ps(values.as_ptr()) }
        }
        #[inline(always)]
        fn mask(bits: &[u32; 8]) -> __m256 {
            unsafe { _mm256_castsi256_ps(_mm256_loadu_si256(bits.as_ptr().cast())) }
        }
        #[inline(always)]
        fn zero() -> __m256 {
            unsafe { _mm256_setzero_ps() }
        }

        #[inline(always)]
        fn add(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_add_ps(a, b) }
        }
        #[inline(always)]
        fn sub(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_sub_ps(a, b) }
        }
        #[inline(always)]
        fn mul(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_mul_ps(a, b) }
        }
        #[inline(always)]
        fn xor(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_xor_ps(a, b) }
        }
        #[inline(always)]
        fn addsub(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_addsub_ps(a, b) }
        }
        #[inline(always)]
        fn fmaddsub(a: __m256, b: __m256, c: __m256) -> __m256 {
            unsafe { _mm256_fmaddsub_ps(a, b, c) }
        }
        #[inline(always)]
        fn fmsubadd(a: __m256, b: __m256, c: __m256) -> __m256 {
            unsafe { _mm256_fmsubadd_ps(a, b, c) }
        }
        #[inline(always)]
        fn fmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
            unsafe { _mm256_fmadd_ps(a, b, c) }
        }

        #[inline(always)]
        fn shufps<const IMM: i32>(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_shuffle_ps::<IMM>(a, b) }
        }
        #[inline(always)]
        fn unpcklpd(a: __m256, b: __m256) -> __m256 {
            unsafe {
                _mm256_castpd_ps(_mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)))
            }
        }
        #[inline(always)]
        fn unpckhpd(a: __m256, b: __m256) -> __m256 {
            unsafe {
                _mm256_castpd_ps(_mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)))
            }
        }
        #[inline(always)]
        fn movsldup(a: __m256) -> __m256 {
            unsafe { _mm256_moveldup_ps(a) }
        }
        #[inline(always)]
        fn movshdup(a: __m256) -> __m256 {
            unsafe { _mm256_movehdup_ps(a) }
        }
        #[inline(always)]
        fn permilps(a: __m256, indices: &[i32; 8]) -> __m256 {
            unsafe { _mm256_permutevar_ps(a, _mm256_loadu_si256(indices.as_ptr().cast())) }
        }
        #[inline(always)]
        fn perm2f128<const IMM: i32>(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_permute2f128_ps::<IMM>(a, b) }
        }
        #[inline(always)]
        fn permpd<const IMM: i32>(a: __m256) -> __m256 {
            unsafe { _mm256_castpd_ps(_mm256_permute4x64_pd::<IMM>(_mm256_castps_pd(a))) }
        }
        #[inline(always)]
        fn blendps<const IMM: i32>(a: __m256, b: __m256) -> __m256 {
            unsafe { _mm256_blend_ps::<IMM>(a, b) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values that make rounding visible: products and sums that are not exact in f32.
    #[cfg(target_arch = "x86_64")]
    fn vector(seed: u32) -> [f32; 8] {
        let mut state = seed;
        std::array::from_fn(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 65_536.0 - 128.0 + 1.0 / 3.0
        })
    }

    #[test]
    fn the_shuffles_follow_the_manual() {
        let a: [f32; 8] = std::array::from_fn(|i| i as f32);
        let b: [f32; 8] = std::array::from_fn(|i| 10.0 + i as f32);
        assert_eq!(
            Portable::shufps::<{ q(1, 0, 1, 0) }>(a, b),
            [0.0, 1.0, 10.0, 11.0, 4.0, 5.0, 14.0, 15.0]
        );
        assert_eq!(
            Portable::shufps::<{ q(2, 3, 0, 1) }>(a, a),
            [1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0]
        );
        assert_eq!(
            Portable::perm2f128::<0x23>(a, b),
            [14.0, 15.0, 16.0, 17.0, 10.0, 11.0, 12.0, 13.0]
        );
        assert_eq!(
            Portable::permpd::<{ q(0, 1, 2, 3) }>(a),
            [6.0, 7.0, 4.0, 5.0, 2.0, 3.0, 0.0, 1.0]
        );
        assert_eq!(
            Portable::blendps::<0b0101_0101>(a, b),
            [10.0, 1.0, 12.0, 3.0, 14.0, 5.0, 16.0, 7.0]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_instructions_and_their_arithmetic_agree() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        #[target_feature(enable = "avx2,fma")]
        fn check(a: [f32; 8], b: [f32; 8], c: [f32; 8]) {
            fn bits<S: Simd>(value: S::V) -> [u32; 8] {
                let mut out = [0f32; 8];
                // SAFETY: 8 floats.
                unsafe { S::store(out.as_mut_ptr(), value) };
                out.map(f32::to_bits)
            }
            fn run<S: Simd>(a: [f32; 8], b: [f32; 8], c: [f32; 8]) -> Vec<[u32; 8]> {
                // SAFETY: arrays of 8 floats.
                let (a, b, c) = unsafe {
                    (
                        S::load(a.as_ptr()),
                        S::load(b.as_ptr()),
                        S::load(c.as_ptr()),
                    )
                };
                let perm = [3, 1, 0, 2, 1, 3, 2, 0];
                vec![
                    bits::<S>(S::add(a, b)),
                    bits::<S>(S::sub(a, b)),
                    bits::<S>(S::mul(a, b)),
                    bits::<S>(S::xor(
                        a,
                        S::mask(&[0x8000_0000, 0, 0, 0x8000_0000, 0, 0, 0, 0]),
                    )),
                    bits::<S>(S::addsub(a, b)),
                    bits::<S>(S::fmaddsub(a, b, c)),
                    bits::<S>(S::fmsubadd(a, b, c)),
                    bits::<S>(S::fmadd(a, b, c)),
                    bits::<S>(S::shufps::<{ q(1, 3, 2, 0) }>(a, b)),
                    bits::<S>(S::unpcklpd(a, b)),
                    bits::<S>(S::unpckhpd(a, b)),
                    bits::<S>(S::movsldup(a)),
                    bits::<S>(S::movshdup(a)),
                    bits::<S>(S::permilps(a, &perm)),
                    bits::<S>(S::perm2f128::<0x31>(a, b)),
                    bits::<S>(S::perm2f128::<0x02>(a, b)),
                    bits::<S>(S::permpd::<{ q(1, 2, 3, 0) }>(a)),
                    bits::<S>(S::blendps::<0b1010_0110>(a, b)),
                ]
            }
            assert_eq!(run::<Avx2>(a, b, c), run::<Portable>(a, b, c));
        }
        for seed in 0..1000 {
            // SAFETY: the processor has avx2 and fma.
            unsafe { check(vector(seed), vector(seed + 7919), vector(seed + 104_729)) };
        }
    }
}
