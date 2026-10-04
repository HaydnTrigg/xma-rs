// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavutil/x86/tx_float.asm
// (FFmpeg n9.1-dev-56-gae4314e2f4); the original files carry these notices:
//
// Copyright (c) Lynne
// Copyright (c) 2026 the xma authors (the Rust port)
//
// xma is free software; you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation; either version
// 2.1 of the License, or (at your option) any later version.
//
// xma is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License (the LICENSE file) for more details.

//! FFmpeg's `fft_sr_asm_float` in its AVX2 build (`libavutil/x86/tx_float.asm`), for the lengths
//! the inverse MDCT of XMA needs: 64, 128 and 256 complex points, input pre-permuted, in place.
//!
//! Written instruction for instruction against [`Simd`], over a file of the sixteen registers
//! named as the assembly names them (`m[0]` is `m0`), so that each macro reads as the original:
//! a value is the same sequence of operations on the same operands as in FFmpeg, which is what
//! makes the output bit-identical. The assembly's `SWAP` renames registers for the text after
//! it; here it swaps their contents at the point the code reaches it, which is the same thing
//! for every value that flows across it. Byte offsets are kept as the assembly has them.
//!
//! The labels are unrolled per length instead of jumped between (`.32pt` falls into `.64pt`,
//! `.128pt` calls `.32pt`), so that everything inlines into one function compiled for AVX2.

use super::simd::{Simd, q};
use crate::tables::{TAB_32, TAB_64, TAB_128, TAB_256};

/// `mmsize`: the bytes of a register.
const MM: usize = 32;

const M_SQRT1_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
// the assembly's literals, each an exact float
#[allow(clippy::excessive_precision)]
const COS16_1: f32 = 0.92387950420379638671875;
#[allow(clippy::excessive_precision)]
const COS16_3: f32 = 0.3826834261417388916015625;
const POS: u32 = 0x0000_0000;
const NEG: u32 = 0x8000_0000;

const D8_MULT_ODD: [f32; 8] = [
    M_SQRT1_2, -M_SQRT1_2, -M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2, -M_SQRT1_2, M_SQRT1_2,
];
const S8_MULT_ODD: [f32; 8] = [
    1.0, 1.0, -1.0, 1.0, -M_SQRT1_2, -M_SQRT1_2, M_SQRT1_2, M_SQRT1_2,
];
const S8_PERM_EVEN: [i32; 8] = [1, 3, 0, 2, 1, 3, 2, 0];
const S8_PERM_ODD1: [i32; 8] = [3, 3, 1, 1, 1, 1, 3, 3];
const S8_PERM_ODD2: [i32; 8] = [1, 2, 0, 3, 1, 0, 0, 1];
const S16_MULT_EVEN: [f32; 8] = [
    1.0, 1.0, M_SQRT1_2, M_SQRT1_2, 1.0, -1.0, M_SQRT1_2, -M_SQRT1_2,
];
const S16_MULT_ODD1: [f32; 8] = [
    COS16_1, COS16_1, COS16_3, COS16_3, COS16_1, -COS16_1, COS16_3, -COS16_3,
];
const S16_MULT_ODD2: [f32; 8] = [
    COS16_3, -COS16_3, COS16_1, -COS16_1, -COS16_3, -COS16_3, -COS16_1, -COS16_1,
];
const S16_PERM: [i32; 8] = [0, 1, 2, 3, 1, 0, 3, 2];
const S16_PERM_BITS: [u32; 8] = [0, 1, 2, 3, 1, 0, 3, 2];

const MASK_MMMMPPPM: [u32; 8] = [NEG, NEG, NEG, NEG, POS, POS, POS, NEG];
const MASK_PPMPMMPM: [u32; 8] = [POS, POS, NEG, POS, NEG, NEG, POS, NEG];
const MASK_MPPMMPMP: [u32; 8] = [NEG, POS, POS, NEG, NEG, POS, NEG, POS];
const MASK_MPMPPMPM: [u32; 8] = [NEG, POS, NEG, POS, POS, NEG, POS, NEG];
const MASK_PMMPPMMP: [u32; 8] = [POS, NEG, NEG, POS, POS, NEG, NEG, POS];
const MASK_PMPMPMPM: [u32; 8] = [POS, NEG, POS, NEG, POS, NEG, POS, NEG];

/// The register file.
pub(crate) type Regs<S> = [<S as Simd>::V; 16];

/// The twiddle tables the transform reads, `ff_tx_tab_<n>_float`: `cos(2 pi i / n)` for
/// `i <= n / 4` (the last one 0), as FFmpeg computes them ([`crate::tables`]).
pub(crate) struct Twiddles {
    pub tab32: &'static [f32],
    pub tab64: &'static [f32],
    pub tab128: &'static [f32],
    pub tab256: &'static [f32],
}

impl Twiddles {
    pub(crate) const FFMPEG: Twiddles = Twiddles {
        tab32: &TAB_32,
        tab64: &TAB_64,
        tab128: &TAB_128,
        tab256: &TAB_256,
    };
}

/// 8 floats at a byte offset.
#[inline(always)]
unsafe fn load<S: Simd>(base: *const f32, byte: usize) -> S::V {
    // SAFETY: the caller's offsets stay inside the buffer (checked by the transform's tests).
    unsafe { S::load(base.add(byte / 4)) }
}

#[inline(always)]
unsafe fn store<S: Simd>(base: *mut f32, byte: usize, value: S::V) {
    // SAFETY: as `load`.
    unsafe { S::store(base.add(byte / 4), value) }
}

#[inline(always)]
unsafe fn store_low<S: Simd>(base: *mut f32, byte: usize, value: S::V) {
    // SAFETY: as `load`.
    unsafe { S::store_low(base.add(byte / 4), value) }
}

#[inline(always)]
unsafe fn store_high<S: Simd>(base: *mut f32, byte: usize, value: S::V) {
    // SAFETY: as `load`.
    unsafe { S::store_high(base.add(byte / 4), value) }
}

/// `FFT4 %1, %2, %3`.
#[inline(always)]
fn fft4<S: Simd>(m: &mut Regs<S>, r1: usize, r2: usize, r3: usize) {
    m[r3] = S::sub(m[r1], m[r2]);
    m[r1] = S::add(m[r1], m[r2]);
    m[r2] = S::shufps::<{ q(1, 0, 1, 0) }>(m[r1], m[r3]);
    m[r1] = S::shufps::<{ q(2, 3, 3, 2) }>(m[r1], m[r3]);
    m[r3] = S::sub(m[r2], m[r1]);
    m[r2] = S::add(m[r2], m[r1]);
    m[r1] = S::shufps::<{ q(1, 0, 1, 0) }>(m[r2], m[r3]);
    m[r2] = S::shufps::<{ q(2, 3, 3, 2) }>(m[r2], m[r3]);
    m[r2] = S::shufps::<{ q(1, 3, 2, 0) }>(m[r2], m[r2]);
}

/// `FFT8 %1, %2, %3, %4, %5, %6`: two 8-point transforms, one per lane.
#[inline(always)]
fn fft8<S: Simd>(
    m: &mut Regs<S>,
    r1: usize,
    r2: usize,
    r3: usize,
    r4: usize,
    r5: usize,
    r6: usize,
) {
    m[r5] = S::add(m[r1], m[r3]);
    m[r6] = S::add(m[r2], m[r4]);
    m[r1] = S::sub(m[r1], m[r3]);
    m[r2] = S::sub(m[r2], m[r4]);
    m[r4] = S::shufps::<{ q(2, 3, 2, 3) }>(m[r1], m[r1]);
    m[r3] = S::shufps::<{ q(3, 0, 3, 2) }>(m[r5], m[r6]);
    m[r1] = S::shufps::<{ q(1, 0, 1, 0) }>(m[r1], m[r1]);
    m[r5] = S::shufps::<{ q(1, 2, 1, 0) }>(m[r5], m[r6]);
    m[r4] = S::xor(m[r4], S::mask(&MASK_PMMPPMMP));
    m[r6] = S::add(m[r5], m[r3]);
    m[r2] = S::mul(m[r2], S::constant(&D8_MULT_ODD));
    m[r5] = S::sub(m[r5], m[r3]);
    m[r3] = S::add(m[r1], m[r4]);
    m[r1] = S::unpcklpd(m[r6], m[r5]);
    m[r4] = S::shufps::<{ q(2, 3, 0, 1) }>(m[r2], m[r2]);
    m[r6] = S::shufps::<{ q(2, 3, 3, 2) }>(m[r6], m[r5]);
    m[r2] = S::addsub(m[r2], m[r4]);
    m[r5] = S::shufps::<{ q(0, 1, 2, 3) }>(m[r2], m[r2]);
    m[r5] = S::addsub(m[r5], m[r2]);
    m[r2] = S::sub(m[r1], m[r6]);
    m[r4] = S::sub(m[r3], m[r5]);
    m[r1] = S::add(m[r1], m[r6]);
    m[r3] = S::add(m[r3], m[r5]);
}

/// `FFT8_AVX %1, %2, %3, %4`.
#[inline(always)]
fn fft8_avx<S: Simd>(m: &mut Regs<S>, r1: usize, r2: usize, r3: usize, r4: usize) {
    m[r3] = S::sub(m[r1], m[r2]);
    m[r1] = S::add(m[r1], m[r2]);
    m[r2] = S::permilps(m[r3], &S8_PERM_ODD1);
    m[r4] = S::shufps::<{ q(3, 3, 2, 2) }>(m[r1], m[r1]);
    m[r3] = S::movsldup(m[r3]);
    m[r1] = S::shufps::<{ q(1, 1, 0, 0) }>(m[r1], m[r1]);
    m[r3] = S::addsub(m[r3], m[r2]);
    m[r1] = S::addsub(m[r1], m[r4]);
    m[r3] = S::mul(m[r3], S::constant(&S8_MULT_ODD));
    m[r1] = S::permilps(m[r1], &S8_PERM_EVEN);
    m[r2] = S::shufps::<{ q(2, 3, 3, 2) }>(m[r3], m[r3]);
    m[r4] = S::xor(m[r1], S::mask(&MASK_MMMMPPPM));
    m[r3] = S::permilps(m[r3], &S8_PERM_ODD2);
    m[r1] = S::perm2f128::<0x03>(m[r1], m[r4]);
    m[r2] = S::addsub(m[r2], m[r3]);
    m[r1] = S::sub(m[r1], m[r4]);
    m[r2] = S::perm2f128::<0x11>(m[r2], m[r2]);
    m[r3] = S::perm2f128::<0x00>(m[r3], m[r3]);
    m[r2] = S::xor(m[r2], S::mask(&MASK_PPMPMMPM));
    m[r2] = S::add(m[r3], m[r2]);
}

/// `FFT16 %1 ... %8` (the eight-register form, FMA3).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fft16<S: Simd>(
    m: &mut Regs<S>,
    r1: usize,
    r2: usize,
    r3: usize,
    r4: usize,
    r5: usize,
    r6: usize,
    r7: usize,
    r8: usize,
) {
    fft4::<S>(m, r3, r4, r5);
    fft8_avx::<S>(m, r1, r2, r6, r7);
    m[r8] = S::mask(&MASK_MPMPPMPM);
    m[r7] = S::mask(&S16_PERM_BITS);
    m[r5] = S::zero();
    m[r6] = S::shufps::<{ q(2, 3, 0, 1) }>(m[r4], m[r4]);
    m[r5] = S::shufps::<{ q(2, 3, 0, 1) }>(m[r5], m[r3]);
    m[r4] = S::mul(m[r4], S::constant(&S16_MULT_ODD1));
    m[r5] = S::xor(m[r5], S::mask(&MASK_MPPMMPMP));
    m[r6] = S::fmadd(m[r6], S::constant(&S16_MULT_ODD2), m[r4]);
    m[r5] = S::add(m[r3], m[r5]);
    m[r5] = S::mul(m[r5], S::constant(&S16_MULT_EVEN));
    m[r4] = S::xor(m[r6], m[r8]);
    m[r3] = S::xor(m[r5], m[r8]);
    m[r4] = S::perm2f128::<0x01>(m[r4], m[r4]);
    m[r3] = S::perm2f128::<0x01>(m[r3], m[r3]);
    m[r6] = S::add(m[r6], m[r4]);
    m[r5] = S::add(m[r5], m[r3]);
    m[r6] = S::permilps(m[r6], &S16_PERM);
    m[r5] = S::permilps(m[r5], &S16_PERM);
    m[r4] = S::sub(m[r2], m[r6]);
    m[r3] = S::add(m[r2], m[r6]);
    m[r2] = S::sub(m[r1], m[r5]);
    m[r1] = S::add(m[r1], m[r5]);
}

/// `SPLIT_RADIX_COMBINE %1 ... %17` (FMA3); `r[0]` is `%2`.
#[inline(always)]
fn combine<S: Simd, const FIRST: bool>(m: &mut Regs<S>, r: [usize; 16]) {
    let [
        a2,
        a3,
        a4,
        a5,
        a6,
        a7,
        a8,
        a9,
        a10,
        a11,
        a12,
        a13,
        a14,
        a15,
        a16,
        a17,
    ] = r;
    if FIRST {
        m[a14] = S::perm2f128::<0x20>(m[a6], m[a7]);
        m[a16] = S::perm2f128::<0x20>(m[a9], m[a8]);
        m[a15] = S::perm2f128::<0x31>(m[a6], m[a7]);
        m[a17] = S::perm2f128::<0x31>(m[a9], m[a8]);
    }
    m[a12] = S::shufps::<{ q(2, 2, 0, 0) }>(m[a10], m[a10]);
    m[a13] = S::shufps::<{ q(1, 1, 3, 3) }>(m[a11], m[a11]);
    m[a10] = S::movshdup(m[a10]);
    m[a11] = S::shufps::<{ q(0, 0, 2, 2) }>(m[a11], m[a11]);
    if FIRST {
        m[a6] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a14], m[a14]);
        m[a8] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a16], m[a16]);
        m[a7] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a15], m[a15]);
        m[a9] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a17], m[a17]);
        m[a14] = S::mul(m[a14], m[a13]);
        m[a16] = S::mul(m[a16], m[a11]);
        m[a15] = S::mul(m[a15], m[a13]);
        m[a17] = S::mul(m[a17], m[a11]);
    } else {
        m[a14] = S::mul(m[a6], m[a13]);
        m[a16] = S::mul(m[a8], m[a11]);
        m[a15] = S::mul(m[a7], m[a13]);
        m[a17] = S::mul(m[a9], m[a11]);
        m[a6] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a6], m[a6]);
        m[a8] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a8], m[a8]);
        m[a7] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a7], m[a7]);
        m[a9] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a9], m[a9]);
    }
    m[a6] = S::fmaddsub(m[a6], m[a12], m[a14]);
    m[a8] = S::fmaddsub(m[a8], m[a10], m[a16]);
    m[a7] = S::fmsubadd(m[a7], m[a12], m[a15]);
    m[a9] = S::fmsubadd(m[a9], m[a10], m[a17]);
    m[a13] = S::mask(&MASK_PMPMPMPM);

    m[a14] = S::add(m[a6], m[a7]);
    m[a16] = S::add(m[a8], m[a9]);
    m[a15] = S::sub(m[a6], m[a7]);
    m[a17] = S::sub(m[a8], m[a9]);

    m[a14] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a14], m[a14]);
    m[a16] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a16], m[a16]);
    m[a15] = S::xor(m[a15], m[a13]);
    m[a17] = S::xor(m[a17], m[a13]);

    m[a6] = S::sub(m[a2], m[a14]);
    m[a8] = S::sub(m[a4], m[a16]);
    m[a7] = S::sub(m[a3], m[a15]);
    m[a9] = S::sub(m[a5], m[a17]);

    m[a2] = S::add(m[a2], m[a14]);
    m[a4] = S::add(m[a4], m[a16]);
    m[a3] = S::add(m[a3], m[a15]);
    m[a5] = S::add(m[a5], m[a17]);
}

/// `SPLIT_RADIX_COMBINE_HALF %1 ... %10` (FMA3); `r[0]` is `%2`.
#[inline(always)]
fn combine_half<S: Simd, const FIRST: bool>(m: &mut Regs<S>, r: [usize; 9]) {
    let [a2, a3, a4, a5, a6, a7, a8, a9, a10] = r;
    if FIRST {
        m[a8] = S::shufps::<{ q(2, 2, 0, 0) }>(m[a6], m[a6]);
        m[a9] = S::shufps::<{ q(1, 1, 3, 3) }>(m[a7], m[a7]);
    } else {
        m[a8] = S::shufps::<{ q(3, 3, 1, 1) }>(m[a6], m[a6]);
        m[a9] = S::shufps::<{ q(0, 0, 2, 2) }>(m[a7], m[a7]);
    }
    m[a10] = S::mul(m[a4], m[a9]);
    m[a9] = S::mul(m[a9], m[a5]);
    m[a4] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a4], m[a4]);
    m[a5] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a5], m[a5]);
    m[a4] = S::fmaddsub(m[a4], m[a8], m[a10]);
    m[a5] = S::fmsubadd(m[a5], m[a8], m[a9]);
    m[a10] = S::mask(&MASK_PMPMPMPM);
    m[a8] = S::add(m[a4], m[a5]);
    m[a9] = S::sub(m[a4], m[a5]);
    m[a8] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a8], m[a8]);
    m[a9] = S::xor(m[a9], m[a10]);
    m[a4] = S::sub(m[a2], m[a8]);
    m[a5] = S::sub(m[a3], m[a9]);
    m[a2] = S::add(m[a2], m[a8]);
    m[a3] = S::add(m[a3], m[a9]);
}

/// `SPLIT_RADIX_COMBINE_LITE %1 ... %9` (FMA3); `r[0]` is `%2`.
#[inline(always)]
fn combine_lite<S: Simd, const FIRST: bool>(m: &mut Regs<S>, r: [usize; 8]) {
    let [a2, a3, a4, a5, a6, a7, a8, a9] = r;
    if FIRST {
        m[a8] = S::shufps::<{ q(2, 2, 0, 0) }>(m[a6], m[a6]);
        m[a9] = S::shufps::<{ q(1, 1, 3, 3) }>(m[a7], m[a7]);
    } else {
        m[a8] = S::shufps::<{ q(3, 3, 1, 1) }>(m[a6], m[a6]);
        m[a9] = S::shufps::<{ q(0, 0, 2, 2) }>(m[a7], m[a7]);
    }
    m[a9] = S::mul(m[a9], m[a4]);
    m[a4] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a4], m[a4]);
    m[a4] = S::fmaddsub(m[a4], m[a8], m[a9]);
    if FIRST {
        m[a9] = S::shufps::<{ q(1, 1, 3, 3) }>(m[a7], m[a7]);
    } else {
        m[a9] = S::shufps::<{ q(0, 0, 2, 2) }>(m[a7], m[a7]);
    }
    m[a9] = S::mul(m[a9], m[a5]);
    m[a5] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a5], m[a5]);
    m[a5] = S::fmsubadd(m[a5], m[a8], m[a9]);
    m[a8] = S::add(m[a4], m[a5]);
    m[a9] = S::sub(m[a4], m[a5]);
    m[a8] = S::shufps::<{ q(2, 3, 0, 1) }>(m[a8], m[a8]);
    m[a9] = S::xor(m[a9], S::mask(&MASK_PMPMPMPM));
    m[a4] = S::sub(m[a2], m[a8]);
    m[a5] = S::sub(m[a3], m[a9]);
    m[a2] = S::add(m[a2], m[a8]);
    m[a3] = S::add(m[a3], m[a9]);
}

// The `.64pt` names (after its `SWAP m4, m1` and `SWAP m6, m3`).
const TX1_E0: usize = 4;
const TX1_E1: usize = 5;
const TX1_O0: usize = 6;
const TX1_O1: usize = 7;
const TX2_E0: usize = 8;
const TX2_E1: usize = 9;
const TX2_O0: usize = 10;
const TX2_O1: usize = 11;
const TW_E: usize = 12;
const TW_O: usize = 13;
const TMP1: usize = 14;
const TMP2: usize = 15;

/// Where the transform is: `inq` and `outq` as byte offsets into the buffer.
struct At {
    buffer: *mut f32,
    inq: usize,
    outq: usize,
}

/// `.32pt` up to its `cmp lenq, 32`: the odd registers stored, `inq` moved on.
#[inline(always)]
unsafe fn pt32<S: Simd>(m: &mut Regs<S>, at: &mut At, tab32: &[f32]) {
    let b = at.buffer;
    // SAFETY (this and every function below): offsets inside a buffer of the transform's
    // length, as the assembly's.
    unsafe {
        m[4] = load::<S>(b, at.inq + 4 * MM);
        m[5] = load::<S>(b, at.inq + 5 * MM);
        m[6] = load::<S>(b, at.inq + 6 * MM);
        m[7] = load::<S>(b, at.inq + 7 * MM);
        fft8::<S>(m, 4, 5, 6, 7, 8, 9);
        m[0] = load::<S>(b, at.inq);
        m[1] = load::<S>(b, at.inq + MM);
        m[2] = load::<S>(b, at.inq + 2 * MM);
        m[3] = load::<S>(b, at.inq + 3 * MM);
        m[8] = load::<S>(tab32.as_ptr(), 0);
        m[9] = S::perm2f128::<0x23>(m[9], load::<S>(tab32.as_ptr(), 32 - 4 * 7));
        fft16::<S>(m, 0, 1, 2, 3, 10, 11, 12, 13);
        combine::<S, true>(m, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        store::<S>(b, at.outq + MM, m[1]);
        store::<S>(b, at.outq + 3 * MM, m[3]);
        store::<S>(b, at.outq + 5 * MM, m[5]);
        store::<S>(b, at.outq + 7 * MM, m[7]);
    }
    at.inq += 8 * MM;
}

/// `.32pt` with `lenq` 32: all of it stored.
#[inline(always)]
unsafe fn pt32_alone<S: Simd>(m: &mut Regs<S>, at: &mut At, tab32: &[f32]) {
    unsafe {
        pt32::<S>(m, at, tab32);
        store::<S>(at.buffer, at.outq, m[0]);
        store::<S>(at.buffer, at.outq + 2 * MM, m[2]);
        store::<S>(at.buffer, at.outq + 4 * MM, m[4]);
        store::<S>(at.buffer, at.outq + 6 * MM, m[6]);
    }
}

/// `.64pt` up to its `cmp tgtq, 64`.
#[inline(always)]
unsafe fn pt64<S: Simd>(m: &mut Regs<S>, at: &mut At, tab64: &[f32]) {
    let b = at.buffer;
    m.swap(4, 1);
    m.swap(6, 3);
    unsafe {
        m[TX1_E0] = load::<S>(b, at.inq);
        m[TX1_E1] = load::<S>(b, at.inq + MM);
        m[TX1_O0] = load::<S>(b, at.inq + 2 * MM);
        m[TX1_O1] = load::<S>(b, at.inq + 3 * MM);
        fft16::<S>(
            m, TX1_E0, TX1_E1, TX1_O0, TX1_O1, TW_E, TW_O, TX2_O0, TX2_O1,
        );
        m[TX2_E0] = load::<S>(b, at.inq + 4 * MM);
        m[TX2_E1] = load::<S>(b, at.inq + 5 * MM);
        m[TX2_O0] = load::<S>(b, at.inq + 6 * MM);
        m[TX2_O1] = load::<S>(b, at.inq + 7 * MM);
        fft16::<S>(m, TX2_E0, TX2_E1, TX2_O0, TX2_O1, TMP1, TMP2, TW_E, TW_O);
        m[TW_E] = load::<S>(tab64.as_ptr(), 0);
        m[TW_O] = S::perm2f128::<0x23>(m[TW_O], load::<S>(tab64.as_ptr(), 64 - 4 * 7));
    }
    at.inq += 8 * MM;
}

/// `SPLIT_RADIX_COMBINE_64`.
#[inline(always)]
unsafe fn combine_64<S: Simd>(m: &mut Regs<S>, at: &At, tab64: &[f32]) {
    let (b, o) = (at.buffer, at.outq);
    unsafe {
        combine_lite::<S, true>(m, [0, 1, TX1_E0, TX2_E0, TW_E, TW_O, TMP1, TMP2]);
        store::<S>(b, o, m[0]);
        store::<S>(b, o + 4 * MM, m[1]);
        store::<S>(b, o + 8 * MM, m[TX1_E0]);
        store::<S>(b, o + 12 * MM, m[TX2_E0]);
        combine_half::<S, false>(m, [2, 3, TX1_O0, TX2_O0, TW_E, TW_O, TMP1, TMP2, 0]);
        store::<S>(b, o + 2 * MM, m[2]);
        store::<S>(b, o + 6 * MM, m[3]);
        store::<S>(b, o + 10 * MM, m[TX1_O0]);
        store::<S>(b, o + 14 * MM, m[TX2_O0]);
        m[TW_E] = load::<S>(tab64.as_ptr(), MM);
        m[TW_O] = S::perm2f128::<0x23>(m[TW_O], load::<S>(tab64.as_ptr(), 64 - 4 * 7 - MM));
        m[0] = load::<S>(b, o + MM);
        m[1] = load::<S>(b, o + 3 * MM);
        m[2] = load::<S>(b, o + 5 * MM);
        m[3] = load::<S>(b, o + 7 * MM);
        combine::<S, false>(
            m,
            [
                0, 2, 1, 3, TX1_E1, TX2_E1, TX1_O1, TX2_O1, TW_E, TW_O, TMP1, TMP2, TX2_O0, TX1_O0,
                TX2_E0, TX1_E0,
            ],
        );
        store::<S>(b, o + MM, m[0]);
        store::<S>(b, o + 3 * MM, m[1]);
        store::<S>(b, o + 5 * MM, m[2]);
        store::<S>(b, o + 7 * MM, m[3]);
        store::<S>(b, o + 9 * MM, m[TX1_E1]);
        store::<S>(b, o + 11 * MM, m[TX1_O1]);
        store::<S>(b, o + 13 * MM, m[TX2_E1]);
        store::<S>(b, o + 15 * MM, m[TX2_O1]);
    }
}

/// `.64pt_deint`: the 64-point transform's last combination, deinterleaved.
#[inline(always)]
unsafe fn pt64_deint<S: Simd>(m: &mut Regs<S>, at: &At, tab64: &[f32]) {
    let (b, o) = (at.buffer, at.outq);
    unsafe {
        combine_lite::<S, true>(m, [0, 1, TX1_E0, TX2_E0, TW_E, TW_O, TMP1, TMP2]);
        combine_half::<S, false>(m, [2, 3, TX1_O0, TX2_O0, TW_E, TW_O, TMP1, TMP2, TW_E]);

        m[TMP1] = S::unpcklpd(m[0], m[2]);
        m[TMP2] = S::unpcklpd(m[1], m[3]);
        m[TW_O] = S::unpcklpd(m[TX1_E0], m[TX1_O0]);
        m[TW_E] = S::unpcklpd(m[TX2_E0], m[TX2_O0]);
        m[0] = S::unpckhpd(m[0], m[2]);
        m[1] = S::unpckhpd(m[1], m[3]);
        m[TX1_E0] = S::unpckhpd(m[TX1_E0], m[TX1_O0]);
        m[TX2_E0] = S::unpckhpd(m[TX2_E0], m[TX2_O0]);

        store_low::<S>(b, o, m[TMP1]);
        store_low::<S>(b, o + 16, m[0]);
        store_low::<S>(b, o + 4 * MM, m[TMP2]);
        store_low::<S>(b, o + 4 * MM + 16, m[1]);

        store_low::<S>(b, o + 8 * MM, m[TW_O]);
        store_low::<S>(b, o + 8 * MM + 16, m[TX1_E0]);
        store_high::<S>(b, o + 9 * MM, m[TW_O]);
        store_high::<S>(b, o + 9 * MM + 16, m[TX1_E0]);

        m[TMP1] = S::perm2f128::<0x31>(m[TMP1], m[0]);
        m[TMP2] = S::perm2f128::<0x31>(m[TMP2], m[1]);

        store_low::<S>(b, o + 12 * MM, m[TW_E]);
        store_low::<S>(b, o + 12 * MM + 16, m[TX2_E0]);
        store_high::<S>(b, o + 13 * MM, m[TW_E]);
        store_high::<S>(b, o + 13 * MM + 16, m[TX2_E0]);

        m[TW_E] = load::<S>(tab64.as_ptr(), MM);
        m[TW_O] = S::perm2f128::<0x23>(m[TW_O], load::<S>(tab64.as_ptr(), 64 - 4 * 7 - MM));

        m[0] = load::<S>(b, o + MM);
        m[1] = load::<S>(b, o + 3 * MM);
        m[2] = load::<S>(b, o + 5 * MM);
        m[3] = load::<S>(b, o + 7 * MM);

        store::<S>(b, o + MM, m[TMP1]);
        store::<S>(b, o + 5 * MM, m[TMP2]);

        combine::<S, false>(
            m,
            [
                0, 2, 1, 3, TX1_E1, TX2_E1, TX1_O1, TX2_O1, TW_E, TW_O, TMP1, TMP2, TX2_O0, TX1_O0,
                TX2_E0, TX1_E0,
            ],
        );

        m[TMP1] = S::unpcklpd(m[0], m[1]);
        m[TMP2] = S::unpcklpd(m[2], m[3]);
        m[TW_E] = S::unpcklpd(m[TX1_E1], m[TX1_O1]);
        m[TW_O] = S::unpcklpd(m[TX2_E1], m[TX2_O1]);
        m[0] = S::unpckhpd(m[0], m[1]);
        m[2] = S::unpckhpd(m[2], m[3]);
        m[TX1_E1] = S::unpckhpd(m[TX1_E1], m[TX1_O1]);
        m[TX2_E1] = S::unpckhpd(m[TX2_E1], m[TX2_O1]);

        store_low::<S>(b, o + 2 * MM, m[TMP1]);
        store_low::<S>(b, o + 2 * MM + 16, m[0]);
        store_high::<S>(b, o + 3 * MM, m[TMP1]);
        store_high::<S>(b, o + 3 * MM + 16, m[0]);

        store_low::<S>(b, o + 6 * MM, m[TMP2]);
        store_low::<S>(b, o + 6 * MM + 16, m[2]);
        store_high::<S>(b, o + 7 * MM, m[TMP2]);
        store_high::<S>(b, o + 7 * MM + 16, m[2]);

        store_low::<S>(b, o + 10 * MM, m[TW_E]);
        store_low::<S>(b, o + 10 * MM + 16, m[TX1_E1]);
        store_high::<S>(b, o + 11 * MM, m[TW_E]);
        store_high::<S>(b, o + 11 * MM + 16, m[TX1_E1]);

        store_low::<S>(b, o + 14 * MM, m[TW_O]);
        store_low::<S>(b, o + 14 * MM + 16, m[TX2_E1]);
        store_high::<S>(b, o + 15 * MM, m[TW_O]);
        store_high::<S>(b, o + 15 * MM + 16, m[TX2_E1]);
    }
}

/// `SPLIT_RADIX_LOAD_COMBINE_4`: `len2`, `len4` and `len6` are the byte offsets of the quarters
/// (`%1`, `%2`, `%3`), `reg` and `tab` the register and table steps (`%4`, `%5`), `offset_*`
/// the extra offsets of the output and of the two table pointers (`%6`, `%7`, `%8`).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn load_combine_4<S: Simd>(
    m: &mut Regs<S>,
    at: &At,
    table: &[f32],
    rtab: usize,
    itab: usize,
    (len2, len4, len6): (usize, usize, usize),
    reg: usize,
    tab: usize,
    (offset_c, offset_r, offset_i): (usize, isize, isize),
) {
    let (b, o) = (at.buffer, at.outq + offset_c);
    let t = table.as_ptr();
    unsafe {
        m[8] = load::<S>(t, (rtab as isize + (tab * MM) as isize + offset_r) as usize);
        m[9] = S::perm2f128::<0x23>(
            m[9],
            load::<S>(t, (itab as isize - (tab * MM) as isize + offset_i) as usize),
        );
        m[0] = load::<S>(b, o + reg * MM);
        m[2] = load::<S>(b, o + (2 + reg) * MM);
        m[1] = load::<S>(b, o + len2 + reg * MM);
        m[3] = load::<S>(b, o + len2 + (2 + reg) * MM);
        m[4] = load::<S>(b, o + len4 + reg * MM);
        m[6] = load::<S>(b, o + len4 + (2 + reg) * MM);
        m[5] = load::<S>(b, o + len6 + reg * MM);
        m[7] = load::<S>(b, o + len6 + (2 + reg) * MM);
        combine::<S, false>(m, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        store::<S>(b, o + reg * MM, m[0]);
        store::<S>(b, o + (2 + reg) * MM, m[2]);
        store::<S>(b, o + len2 + reg * MM, m[1]);
        store::<S>(b, o + len2 + (2 + reg) * MM, m[3]);
        store::<S>(b, o + len4 + reg * MM, m[4]);
        store::<S>(b, o + len4 + (2 + reg) * MM, m[6]);
        store::<S>(b, o + len6 + reg * MM, m[5]);
        store::<S>(b, o + len6 + (2 + reg) * MM, m[7]);
    }
}

/// `SPLIT_RADIX_LOAD_COMBINE_FULL 2*len, 6*len[, offsets]`.
#[inline(always)]
unsafe fn load_combine_full<S: Simd>(
    m: &mut Regs<S>,
    at: &At,
    table: &[f32],
    (rtab, itab): (usize, usize),
    len: usize,
    offsets: (usize, isize, isize),
) {
    let quarters = (2 * len, 4 * len, 6 * len);
    unsafe {
        load_combine_4::<S>(m, at, table, rtab, itab, quarters, 0, 0, offsets);
        load_combine_4::<S>(m, at, table, rtab, itab, quarters, 1, 1, offsets);
        load_combine_4::<S>(m, at, table, rtab, itab, quarters, 4, 2, offsets);
        load_combine_4::<S>(m, at, table, rtab, itab, quarters, 5, 3, offsets);
    }
}

/// `SPLIT_RADIX_COMBINE_DEINTERLEAVE_2 %1, %2, %3, %4, %5, 0`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn combine_deinterleave_2<S: Simd>(
    m: &mut Regs<S>,
    at: &At,
    table: &[f32],
    rtab: usize,
    itab: usize,
    reg: usize,
    tab: usize,
    (q2, q4, q6): (usize, usize, usize),
) {
    let (b, o) = (at.buffer, at.outq);
    let t = table.as_ptr();
    unsafe {
        m[8] = load::<S>(t, rtab + tab * MM);
        m[9] = S::perm2f128::<0x23>(m[9], load::<S>(t, itab - tab * MM));
        m[0] = load::<S>(b, o + reg * MM);
        m[2] = load::<S>(b, o + (2 + reg) * MM);
        m[1] = load::<S>(b, o + q2 + reg * MM);
        m[3] = load::<S>(b, o + q2 + (2 + reg) * MM);
        m[4] = load::<S>(b, o + q4 + reg * MM);
        m[6] = load::<S>(b, o + q4 + (2 + reg) * MM);
        m[5] = load::<S>(b, o + q6 + reg * MM);
        m[7] = load::<S>(b, o + q6 + (2 + reg) * MM);

        combine::<S, false>(m, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        m[10] = S::unpckhpd(m[0], m[2]);
        m[11] = S::unpckhpd(m[1], m[3]);
        m[12] = S::unpckhpd(m[4], m[6]);
        m[13] = S::unpckhpd(m[5], m[7]);
        m[0] = S::unpcklpd(m[0], m[2]);
        m[1] = S::unpcklpd(m[1], m[3]);
        m[4] = S::unpcklpd(m[4], m[6]);
        m[5] = S::unpcklpd(m[5], m[7]);

        store_low::<S>(b, o + reg * MM, m[0]);
        store_low::<S>(b, o + reg * MM + 16, m[10]);
        store_low::<S>(b, o + q2 + reg * MM, m[1]);
        store_low::<S>(b, o + q2 + reg * MM + 16, m[11]);
        store_low::<S>(b, o + q4 + reg * MM, m[4]);
        store_low::<S>(b, o + q4 + reg * MM + 16, m[12]);
        store_low::<S>(b, o + q6 + reg * MM, m[5]);
        store_low::<S>(b, o + q6 + reg * MM + 16, m[13]);

        m[10] = S::perm2f128::<0x13>(m[10], m[0]);
        m[11] = S::perm2f128::<0x13>(m[11], m[1]);
        m[12] = S::perm2f128::<0x13>(m[12], m[4]);
        m[13] = S::perm2f128::<0x13>(m[13], m[5]);

        m[8] = load::<S>(t, rtab + (1 + tab) * MM);
        m[9] = S::perm2f128::<0x23>(m[9], load::<S>(t, itab - (1 + tab) * MM));

        m[0] = load::<S>(b, o + (1 + reg) * MM);
        m[2] = load::<S>(b, o + (3 + reg) * MM);
        m[1] = load::<S>(b, o + q2 + (1 + reg) * MM);
        m[3] = load::<S>(b, o + q2 + (3 + reg) * MM);

        store::<S>(b, o + (1 + reg) * MM, m[10]);
        store::<S>(b, o + q2 + (1 + reg) * MM, m[11]);

        m[4] = load::<S>(b, o + q4 + (1 + reg) * MM);
        m[6] = load::<S>(b, o + q4 + (3 + reg) * MM);
        m[5] = load::<S>(b, o + q6 + (1 + reg) * MM);
        m[7] = load::<S>(b, o + q6 + (3 + reg) * MM);

        store::<S>(b, o + q4 + (1 + reg) * MM, m[12]);
        store::<S>(b, o + q6 + (1 + reg) * MM, m[13]);

        combine::<S, false>(m, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);

        m[8] = S::unpcklpd(m[0], m[2]);
        m[9] = S::unpcklpd(m[1], m[3]);
        m[10] = S::unpcklpd(m[4], m[6]);
        m[11] = S::unpcklpd(m[5], m[7]);
        m[0] = S::unpckhpd(m[0], m[2]);
        m[1] = S::unpckhpd(m[1], m[3]);
        m[4] = S::unpckhpd(m[4], m[6]);
        m[5] = S::unpckhpd(m[5], m[7]);

        store_low::<S>(b, o + (2 + reg) * MM, m[8]);
        store_low::<S>(b, o + (2 + reg) * MM + 16, m[0]);
        store_high::<S>(b, o + (3 + reg) * MM, m[8]);
        store_high::<S>(b, o + (3 + reg) * MM + 16, m[0]);

        store_low::<S>(b, o + q2 + (2 + reg) * MM, m[9]);
        store_low::<S>(b, o + q2 + (2 + reg) * MM + 16, m[1]);
        store_high::<S>(b, o + q2 + (3 + reg) * MM, m[9]);
        store_high::<S>(b, o + q2 + (3 + reg) * MM + 16, m[1]);

        store_low::<S>(b, o + q4 + (2 + reg) * MM, m[10]);
        store_low::<S>(b, o + q4 + (2 + reg) * MM + 16, m[4]);
        store_high::<S>(b, o + q4 + (3 + reg) * MM, m[10]);
        store_high::<S>(b, o + q4 + (3 + reg) * MM + 16, m[4]);

        store_low::<S>(b, o + q6 + (2 + reg) * MM, m[11]);
        store_low::<S>(b, o + q6 + (2 + reg) * MM + 16, m[5]);
        store_high::<S>(b, o + q6 + (3 + reg) * MM, m[11]);
        store_high::<S>(b, o + q6 + (3 + reg) * MM + 16, m[5]);
    }
}

/// `.deinterleave` for a transform of `len` points, `table` the `len`-point twiddles.
#[inline(always)]
unsafe fn deinterleave<S: Simd>(m: &mut Regs<S>, at: &mut At, table: &[f32], len: usize) {
    let quarters = (2 * len, 4 * len, 6 * len);
    let mut rtab = 0usize;
    let mut itab = len - 4 * 7;
    let mut target = len as isize;
    loop {
        unsafe {
            combine_deinterleave_2::<S>(m, at, table, rtab, itab, 0, 0, quarters);
            combine_deinterleave_2::<S>(m, at, table, rtab, itab, 4, 2, quarters);
        }
        at.outq += 8 * MM;
        rtab += 4 * MM;
        itab = itab.wrapping_sub(4 * MM);
        target -= (4 * MM) as isize;
        if target <= 0 {
            break;
        }
    }
}

/// `.128pt` up to its `cmp tgtq, 128`: two 32-point transforms after the 64-point one.
#[inline(always)]
unsafe fn pt128_halves<S: Simd>(m: &mut Regs<S>, at: &mut At, twiddles: &Twiddles) {
    at.outq += 16 * MM;
    unsafe { pt32_alone::<S>(m, at, twiddles.tab32) };
    at.outq += 8 * MM;
    unsafe { pt32_alone::<S>(m, at, twiddles.tab32) };
    at.outq -= 24 * MM;
}

/// The whole `.64pt` of a transform longer than 64 points (`lenq` 64 or more): the 32 points
/// before it, the combination after it.
#[inline(always)]
unsafe fn pt64_combined<S: Simd>(m: &mut Regs<S>, at: &mut At, twiddles: &Twiddles) {
    unsafe {
        pt32::<S>(m, at, twiddles.tab32);
        pt64::<S>(m, at, twiddles.tab64);
        combine_64::<S>(m, at, twiddles.tab64);
    }
}

/// The transform of `len` (64, 128 or 256) complex points held in `buffer` (pre-permuted by
/// [`super::parity_revtab`]), in place, the output in natural order.
///
/// # Safety
///
/// `buffer` holds `2 * len` floats; with [`super::simd::Avx2`], the processor has AVX2 and FMA.
#[inline(always)]
pub(crate) unsafe fn transform<S: Simd>(buffer: &mut [f32], len: usize, twiddles: &Twiddles) {
    assert!(matches!(len, 64 | 128 | 256) && buffer.len() >= 2 * len);
    let mut m: Regs<S> = [S::zero(); 16];
    let mut at = At {
        buffer: buffer.as_mut_ptr(),
        inq: 0,
        outq: 0,
    };
    unsafe {
        match len {
            64 => {
                pt32::<S>(&mut m, &mut at, twiddles.tab32);
                pt64::<S>(&mut m, &mut at, twiddles.tab64);
                pt64_deint::<S>(&mut m, &at, twiddles.tab64);
            }
            128 => {
                pt64_combined::<S>(&mut m, &mut at, twiddles);
                pt128_halves::<S>(&mut m, &mut at, twiddles);
                deinterleave::<S>(&mut m, &mut at, twiddles.tab128, 128);
            }
            _ => {
                pt64_combined::<S>(&mut m, &mut at, twiddles);
                pt128_halves::<S>(&mut m, &mut at, twiddles);
                load_combine_full::<S>(
                    &mut m,
                    &at,
                    twiddles.tab128,
                    (0, 128 - 4 * 7),
                    128,
                    (0, 0, 0),
                );
                // .256pt: two 64-point transforms
                at.outq += 32 * MM;
                pt64_combined::<S>(&mut m, &mut at, twiddles);
                at.outq += 16 * MM;
                pt64_combined::<S>(&mut m, &mut at, twiddles);
                at.outq -= 48 * MM;
                deinterleave::<S>(&mut m, &mut at, twiddles.tab256, 256);
            }
        }
    }
}
