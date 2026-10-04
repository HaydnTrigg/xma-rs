// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavutil/tx_template.c
// (FFmpeg n9.1-dev-56-gae4314e2f4); the original files carry these notices:
//
// Copyright (c) Lynne
// Copyright (c) 2008 Loren Merritt
// Copyright (c) 2002 Fabrice Bellard
// Copyright (c) 2026 the xma authors (the Rust port)
//
// xma is free software; you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation; either version
// 2.1 of the License, or (at your option) any later version.
//
// xma is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License (the LICENSE file) for more details.

use super::*;

/// Coefficients like a decoder's: integers times a band's quantizer, of every size.
fn coefficients(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let value = ((state >> 16) % 2001) as f32 - 1000.0;
            value * (1.0 + (i % 29) as f32 / 7.0)
        })
        .collect()
}

fn run(kernel: Kernel, len: usize, input: &[f32]) -> Vec<f32> {
    let mut out = vec![f32::NAN; len];
    inverse(kernel, len, &mut out, input);
    out
}

/// `ff_tx_mdct_naive_inv`, in double precision.
fn reference(len: usize, input: &[f32]) -> Vec<f64> {
    let half = len / 2;
    let scale = f64::from((1.0 / half as f64 / 32768.0) as f32);
    let phase = std::f64::consts::PI / (4.0 * len as f64);
    let mut out = vec![0.0; len];
    for i in 0..half {
        let (mut down, mut up) = (0.0f64, 0.0f64);
        let i_d = phase * (4 * half - 2 * i - 1) as f64;
        let i_u = phase * (3 * len + 2 * i + 1) as f64;
        for (j, &value) in input.iter().enumerate().take(len) {
            let a = (2 * j + 1) as f64;
            down += (a * i_d).cos() * f64::from(value);
            up += (a * i_u).cos() * f64::from(value);
        }
        out[i] = down * scale;
        out[i + half] = -up * scale;
    }
    out
}

#[test]
fn the_fft_input_order_is_a_permutation() {
    for len in [64, 128, 256] {
        let mut map = parity_revtab(len);
        map.sort_unstable();
        assert_eq!(map, (0..len as i32).collect::<Vec<_>>());
    }
}

#[test]
fn the_transform_is_the_inverse_mdct() {
    for len in [128, 256, 512] {
        for seed in 0..4 {
            let input = coefficients(len, seed);
            let out = run(Kernel::Portable, len, &input);
            let expected = reference(len, &input);
            let peak = expected
                .iter()
                .fold(0.0f64, |peak, value| peak.max(value.abs()));
            for (index, (&got, &want)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (f64::from(got) - want).abs() <= peak * 2e-6,
                    "len {len} seed {seed} sample {index}: {got} against {want}"
                );
            }
        }
    }
}

#[test]
fn the_kernels_agree_to_the_bit() {
    if !Kernel::Avx2.is_available() {
        return;
    }
    for len in [128, 256, 512] {
        for seed in 0..64 {
            let input = coefficients(len, seed);
            let avx2: Vec<u32> = run(Kernel::Avx2, len, &input)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let portable: Vec<u32> = run(Kernel::Portable, len, &input)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(avx2, portable, "len {len} seed {seed}");
        }
    }
}
