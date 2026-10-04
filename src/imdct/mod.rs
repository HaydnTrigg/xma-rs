// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavutil/tx.c,
// libavutil/tx_template.c and libavutil/x86/tx_float.asm (FFmpeg n9.1-dev-56-gae4314e2f4); the
// original files carry these notices:
//
// Copyright (c) Lynne
// Copyright (c) 2008 Loren Merritt
// Copyright (c) 2002 Fabrice Bellard
// Copyright (c) the FFmpeg developers
// Copyright (c) 2026 the xma authors (the Rust port)
//
// xma is free software; you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation; either version
// 2.1 of the License, or (at your option) any later version.
//
// xma is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License (the LICENSE file) for more details.

//! The inverse MDCT of the WMA Pro decoder: FFmpeg's `av_tx` `AV_TX_FLOAT_MDCT`, inverse, as an
//! x86-64 build with AVX2 runs it (`mdct_inv_float_avx2` over `fft_sr_asm_float_avx2`, the
//! codelets `av_tx_init` picks on such a processor), for the three lengths XMA uses (128, 256
//! and 512 coefficients). It is the "half" inverse MDCT: `n` coefficients in, `n` samples out.
//!
//! The arithmetic is FFmpeg's to the bit, FMA included, and it does not depend on the processor
//! this runs on: [`Kernel::Avx2`] runs the instructions, [`Kernel::Portable`] their exact
//! arithmetic ([`simd`]). FFmpeg itself gives other bits on a processor without AVX2 (its C
//! transform rounds differently); this port always gives the AVX2 ones.

mod fft;
pub(crate) mod simd;

use std::sync::OnceLock;

use crate::tables::{MDCT_EXP_128, MDCT_EXP_256, MDCT_EXP_512};
use fft::Twiddles;
use simd::{Portable, Simd, q};

/// How the transforms run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    /// The AVX2 and FMA instructions (x86-64 processors that have them).
    Avx2,
    /// Their arithmetic in portable code: the same results, slower.
    Portable,
}

impl Kernel {
    /// The fastest kernel this processor runs.
    pub fn detect() -> Kernel {
        if Kernel::Avx2.is_available() {
            Kernel::Avx2
        } else {
            Kernel::Portable
        }
    }

    /// Whether this processor runs the kernel.
    pub fn is_available(self) -> bool {
        match self {
            Kernel::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(not(target_arch = "x86_64"))]
            Kernel::Avx2 => false,
        }
    }
}

/// `split_radix_permutation` (`libavutil/tx.c`).
fn split_radix_permutation(i: i32, len: i32, inverse: i32) -> i32 {
    let len = len >> 1;
    if len <= 1 {
        return i & 1;
    }
    if i & len == 0 {
        return split_radix_permutation(i, len, inverse) * 2;
    }
    let len = len >> 1;
    split_radix_permutation(i, len, inverse) * 4 + 1 - 2 * (i32::from(i & len == 0) ^ inverse)
}

/// `parity_revtab_generator` (`libavutil/tx.c`).
#[allow(clippy::too_many_arguments)]
fn parity_revtab_generator(
    revtab: &mut [i32],
    n: i32,
    inverse: i32,
    offset: i32,
    is_dual: bool,
    dual_high: bool,
    len: i32,
    basis: i32,
    dual_stride: i32,
    inverse_lookup: bool,
) {
    let len = len >> 1;
    if len <= basis {
        let is_dual = is_dual && dual_stride != 0;
        let dual_high = i32::from(is_dual && dual_high);
        let stride = if is_dual { dual_stride.min(len) } else { 0 };
        let mut even = offset + dual_high * (stride - 2 * len);
        let mut odd = even + len + i32::from(is_dual && dual_high == 0) * len + dual_high * len;
        for i in 0..len {
            let k1 = -split_radix_permutation(offset + i * 2, n, inverse) & (n - 1);
            let k2 = -split_radix_permutation(offset + i * 2 + 1, n, inverse) & (n - 1);
            if inverse_lookup {
                revtab[even as usize] = k1;
                revtab[odd as usize] = k2;
            } else {
                revtab[k1 as usize] = even;
                revtab[k2 as usize] = odd;
            }
            even += 1;
            odd += 1;
            if stride != 0 && (i + 1) % stride == 0 {
                even += stride;
                odd += stride;
            }
        }
        return;
    }
    let args = (n, inverse, basis, dual_stride, inverse_lookup);
    let recurse = |revtab: &mut [i32], offset, is_dual, dual_high, len| {
        parity_revtab_generator(
            revtab, args.0, args.1, offset, is_dual, dual_high, len, args.2, args.3, args.4,
        )
    };
    recurse(revtab, offset, false, false, len);
    recurse(revtab, offset + len, true, false, len >> 1);
    recurse(revtab, offset + len + (len >> 1), true, true, len >> 1);
}

/// The input order of the AVX2 split-radix FFT of `len` points
/// (`ff_tx_gen_split_radix_parity_revtab(s, len, inverse, gather, 8, 2)`): its input `i` is the
/// natural input `map[i]`.
fn parity_revtab(len: usize) -> Vec<i32> {
    let mut map = vec![0i32; len];
    parity_revtab_generator(
        &mut map, len as i32, 1, 0, false, false, len as i32, 4, 2, true,
    );
    map
}

/// One length's tables (`m_inv_init` and `ff_tx_mdct_gen_exp_float`).
struct Mdct {
    /// Coefficients.
    len: usize,
    /// Where the pre-rotation puts complex value `j`: the inverse of the FFT's input order.
    scatter: Vec<i32>,
    /// The rotation factors, `len / 2` complex values in natural order
    /// (`(cos, sin)(pi / 2 * (i + 1 / 8) / (len / 2)) * sqrt(scale)`, [`crate::tables`]).
    exp: &'static [f32],
}

impl Mdct {
    /// The transform of `len` coefficients with the decoder's scale: `1 / (len / 2) / 32768`,
    /// which also turns 16-bit sample values into [-1, 1].
    fn new(len: usize, exp: &'static [f32]) -> Mdct {
        assert_eq!(exp.len(), len);
        let map = parity_revtab(len / 2);
        let mut scatter = vec![0i32; len / 2];
        for (i, &at) in map.iter().enumerate() {
            scatter[at as usize] = i as i32;
        }
        Mdct { len, scatter, exp }
    }
}

/// Every table, made once.
struct Tables {
    twiddles: Twiddles,
    mdct: [Mdct; 3],
}

fn tables() -> &'static Tables {
    static TABLES: OnceLock<Tables> = OnceLock::new();
    TABLES.get_or_init(|| Tables {
        twiddles: Twiddles::FFMPEG,
        mdct: [
            Mdct::new(128, &MDCT_EXP_128),
            Mdct::new(256, &MDCT_EXP_256),
            Mdct::new(512, &MDCT_EXP_512),
        ],
    })
}

/// `mdct_inv_float` (the `.stride4` path: the input is contiguous).
///
/// # Safety
///
/// With [`simd::Avx2`], the processor has AVX2 and FMA.
#[inline(always)]
unsafe fn inverse_with<S: Simd>(mdct: &Mdct, twiddles: &Twiddles, out: &mut [f32], input: &[f32]) {
    let n = mdct.len;
    assert!(out.len() >= n && input.len() >= n);
    let (input_at, exp_at) = (input.as_ptr(), mdct.exp.as_ptr());
    for k in 0..n / 16 {
        // SAFETY: 8 floats from 8k and from n - 8 - 8k, inside both arrays.
        let (m4, m3, m2, m5) = unsafe {
            (
                S::load(input_at.add(8 * k)),
                S::load(input_at.add(n - 8 - 8 * k)),
                S::load(exp_at.add(8 * k)),
                S::load(exp_at.add(n - 8 - 8 * k)),
            )
        };
        let m1 = S::movsldup(m4);
        let m0 = S::movshdup(m3);
        let m4 = S::movshdup(m4);
        let m3 = S::movsldup(m3);
        let m0 = S::permpd::<{ q(0, 1, 2, 3) }>(m0);
        let m7 = S::shufps::<{ q(2, 3, 0, 1) }>(m2, m2);
        let m4 = S::permpd::<{ q(0, 1, 2, 3) }>(m4);
        let m8 = S::shufps::<{ q(2, 3, 0, 1) }>(m5, m5);
        let m1 = S::mul(m1, m7);
        let m3 = S::mul(m3, m8);
        let m0 = S::fmaddsub(m0, m2, m1);
        let m4 = S::fmaddsub(m4, m5, m3);
        // the scatter into the FFT's input order
        let (mut low, mut high) = ([0f32; 8], [0f32; 8]);
        // SAFETY: 8 floats each.
        unsafe {
            S::store(low.as_mut_ptr(), m0);
            S::store(high.as_mut_ptr(), m4);
        }
        for c in 0..4 {
            let at = mdct.scatter[4 * k + c] as usize * 2;
            out[at] = low[2 * c];
            out[at + 1] = low[2 * c + 1];
            let at = mdct.scatter[n / 2 - 4 - 4 * k + c] as usize * 2;
            out[at] = high[2 * c];
            out[at + 1] = high[2 * c + 1];
        }
    }
    // SAFETY: n floats of n / 2 complex points; the caller's processor.
    unsafe { fft::transform::<S>(&mut out[..n], n / 2, twiddles) };
    let out_at = out.as_mut_ptr();
    for i in 0..n / 16 {
        let (low, high) = (8 * i, n - 8 - 8 * i);
        // SAFETY: 8 floats at both ends, inside both arrays; both read before either is written.
        unsafe {
            let m2 = S::load(exp_at.add(high));
            let m3 = S::load(exp_at.add(low));
            let m0 = S::load(out_at.add(high));
            let m1 = S::load(out_at.add(low));
            let m4 = S::movshdup(m2);
            let m5 = S::movshdup(m3);
            let m6 = S::movsldup(m2);
            let m7 = S::movsldup(m3);
            let m2 = S::shufps::<{ q(2, 3, 0, 1) }>(m0, m0);
            let m3 = S::shufps::<{ q(2, 3, 0, 1) }>(m1, m1);
            let m6 = S::mul(m6, m0);
            let m7 = S::mul(m7, m1);
            let m4 = S::fmaddsub(m4, m2, m6);
            let m5 = S::fmaddsub(m5, m3, m7);
            let m3 = S::permpd::<{ q(0, 1, 2, 3) }>(m5);
            let m2 = S::permpd::<{ q(0, 1, 2, 3) }>(m4);
            let m1 = S::blendps::<0b0101_0101>(m2, m5);
            let m0 = S::blendps::<0b0101_0101>(m3, m4);
            S::store(out_at.add(high), m0);
            S::store(out_at.add(low), m1);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn inverse_avx2(mdct: &Mdct, twiddles: &Twiddles, out: &mut [f32], input: &[f32]) {
    // SAFETY: compiled for, and only called on, a processor with AVX2 and FMA.
    unsafe { inverse_with::<simd::Avx2>(mdct, twiddles, out, input) }
}

fn inverse_portable(mdct: &Mdct, twiddles: &Twiddles, out: &mut [f32], input: &[f32]) {
    // SAFETY: the portable instructions need no processor feature.
    unsafe { inverse_with::<Portable>(mdct, twiddles, out, input) }
}

/// The inverse MDCT of `len` (128, 256 or 512) coefficients of `input` into `len` samples of
/// `out`. `kernel` must be available ([`Kernel::is_available`]).
pub(crate) fn inverse(kernel: Kernel, len: usize, out: &mut [f32], input: &[f32]) {
    let tables = tables();
    let mdct = match len {
        128 => &tables.mdct[0],
        256 => &tables.mdct[1],
        512 => &tables.mdct[2],
        _ => panic!("an inverse MDCT of {len} coefficients"),
    };
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => {
            assert!(Kernel::Avx2.is_available());
            // SAFETY: the processor has AVX2 and FMA (checked above).
            unsafe { inverse_avx2(mdct, &tables.twiddles, out, input) }
        }
        _ => inverse_portable(mdct, &tables.twiddles, out, input),
    }
}

#[cfg(test)]
mod tests;
