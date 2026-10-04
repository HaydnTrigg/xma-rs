// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavcodec/vlc.c (FFmpeg
// n9.1-dev-56-gae4314e2f4); the original files carry these notices:
//
// Copyright (c) 2000, 2001 Fabrice Bellard
// Copyright (c) 2002-2004 Michael Niedermayer <michaelni@gmx.at>
// Copyright (c) 2010 Loren Merritt
// Copyright (c) 2026 the xma authors (the Rust port)
//
// xma is free software; you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation; either version
// 2.1 of the License, or (at your option) any later version.
//
// xma is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License (the LICENSE file) for more details.

//! FFmpeg's VLC tables (`libavcodec/vlc.c`, `ff_vlc_init_from_lengths`): codes assigned from
//! their lengths in the order listed, looked up by a root table of `bits` bits and subtables
//! for the longer codes. Built the same way, entry for entry, so a code FFmpeg's tables do not
//! hold reads the same here ([`crate::bits::GetBits::vlc`]).

use std::sync::OnceLock;

use crate::tables::{
    COEF0_LENS, COEF0_SYMS, COEF1_TABLE, SCALE_RL_TABLE, SCALE_TABLE, VEC1_TABLE, VEC2_TABLE,
    VEC4_LENS, VEC4_SYMS,
};

/// `VLCElem`: a symbol and its length; a negative length is a subtable of that many bits whose
/// entries start at the symbol.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VlcElem {
    pub sym: i16,
    pub len: i16,
}

/// `VLCcode`.
#[derive(Clone, Copy, Debug)]
struct Code {
    bits: i32,
    symbol: i16,
    /// Left aligned.
    code: u32,
}

/// `ff_vlc_init_from_lengths`: `lengths[i]` bits for `symbols[i] + offset`. A negative length
/// takes code space without a symbol.
fn from_lengths(bits: i32, lengths: &[i8], symbols: &[i32], offset: i32) -> Vec<VlcElem> {
    let length_max = 32.min(3 * bits);
    let mut codes = Vec::with_capacity(lengths.len());
    let mut code = 0u64;
    for (&length, &symbol) in lengths.iter().zip(symbols) {
        let length = match length {
            0 => continue,
            length if length > 0 => {
                codes.push(Code {
                    bits: i32::from(length),
                    symbol: (symbol + offset) as i16,
                    code: code as u32,
                });
                i32::from(length)
            }
            length => -i32::from(length),
        };
        assert!(
            length <= length_max && code & ((1u64 << (32 - length)) - 1) == 0,
            "an invalid VLC length {length}"
        );
        code += 1u64 << (32 - length);
        assert!(code <= 1u64 << 32, "an overdetermined VLC tree");
    }
    let mut table = Vec::new();
    build_table(&mut table, bits, &mut codes);
    table
}

/// `build_table`: the table of `table_bits` bits for `codes` appended to `table`; its index.
fn build_table(table: &mut Vec<VlcElem>, table_bits: i32, codes: &mut [Code]) -> usize {
    let size = 1usize << table_bits;
    let start = table.len();
    table.resize(start + size, VlcElem::default());
    let mut i = 0;
    while i < codes.len() {
        let n = codes[i].bits;
        let code = codes[i].code;
        let symbol = codes[i].symbol;
        if n <= table_bits {
            let first = (code >> (32 - table_bits)) as usize;
            for entry in &mut table[start + first..start + first + (1 << (table_bits - n))] {
                assert!(
                    !((entry.len != 0 || entry.sym != 0)
                        && (i32::from(entry.len) != n || entry.sym != symbol)),
                    "incorrect VLC codes"
                );
                entry.len = n as i16;
                entry.sym = symbol;
            }
            i += 1;
        } else {
            // the codes that share this prefix go to a subtable
            let prefix = code >> (32 - table_bits);
            let mut subtable_bits = n - table_bits;
            codes[i].bits = n - table_bits;
            codes[i].code = code << table_bits;
            let mut k = i + 1;
            while k < codes.len() {
                let n = codes[k].bits - table_bits;
                if n <= 0 || codes[k].code >> (32 - table_bits) != prefix {
                    break;
                }
                codes[k].bits = n;
                codes[k].code <<= table_bits;
                subtable_bits = subtable_bits.max(n);
                k += 1;
            }
            let subtable_bits = subtable_bits.min(table_bits);
            table[start + prefix as usize].len = -subtable_bits as i16;
            let index = build_table(table, subtable_bits, &mut codes[i..k]);
            table[start + prefix as usize].sym = index as i16;
            assert_eq!(
                table[start + prefix as usize].sym as usize,
                index,
                "strange codes"
            );
            i = k;
        }
    }
    for entry in &mut table[start..start + size] {
        if entry.len == 0 {
            entry.sym = -1;
        }
    }
    start
}

/// The decoder's codebooks (`decode_init_static`).
pub(crate) struct Codebooks {
    /// The DPCM scale factors (8-bit root, up to 3 levels).
    pub scale: Vec<VlcElem>,
    /// The run-level scale factors (9-bit root, up to 3 levels).
    pub scale_rl: Vec<VlcElem>,
    /// The run-level coefficients, by table (9-bit root, up to 3 levels).
    pub coef: [Vec<VlcElem>; 2],
    /// The vector coded coefficients, 4, 2 and 1 per symbol (9-bit root, up to 2 levels).
    pub vec4: Vec<VlcElem>,
    pub vec2: Vec<VlcElem>,
    pub vec1: Vec<VlcElem>,
}

/// Root bits of every table but the scale factors' (`VLCBITS`).
pub(crate) const VLC_BITS: u32 = 9;
/// Root bits of the scale factors' (`SCALEVLCBITS`).
pub(crate) const SCALE_VLC_BITS: u32 = 8;

fn lengths_of(pairs: &[[u8; 2]]) -> (Vec<i8>, Vec<i32>) {
    pairs
        .iter()
        .map(|&[symbol, length]| (length as i8, i32::from(symbol)))
        .unzip()
}

fn build() -> Codebooks {
    let (scale_lengths, scale_symbols) = lengths_of(&SCALE_TABLE);
    let (scale_rl_lengths, scale_rl_symbols) = lengths_of(&SCALE_RL_TABLE);
    let (coef1_lengths, coef1_symbols) = lengths_of(&COEF1_TABLE);
    let (vec2_lengths, vec2_symbols) = lengths_of(&VEC2_TABLE);
    let (vec1_lengths, vec1_symbols) = lengths_of(&VEC1_TABLE);
    let coef0_lengths: Vec<i8> = COEF0_LENS.iter().map(|&length| length as i8).collect();
    let coef0_symbols: Vec<i32> = COEF0_SYMS.iter().map(|&symbol| i32::from(symbol)).collect();
    let vec4_lengths: Vec<i8> = VEC4_LENS.iter().map(|&length| length as i8).collect();
    let vec4_symbols: Vec<i32> = VEC4_SYMS.iter().map(|&symbol| i32::from(symbol)).collect();
    let bits = VLC_BITS as i32;
    Codebooks {
        scale: from_lengths(SCALE_VLC_BITS as i32, &scale_lengths, &scale_symbols, -60),
        scale_rl: from_lengths(bits, &scale_rl_lengths, &scale_rl_symbols, 0),
        coef: [
            from_lengths(bits, &coef0_lengths, &coef0_symbols, 0),
            from_lengths(bits, &coef1_lengths, &coef1_symbols, 0),
        ],
        vec4: from_lengths(bits, &vec4_lengths, &vec4_symbols, -1),
        vec2: from_lengths(bits, &vec2_lengths, &vec2_symbols, -1),
        vec1: from_lengths(bits, &vec1_lengths, &vec1_symbols, 0),
    }
}

/// The codebooks, built once.
pub(crate) fn codebooks() -> &'static Codebooks {
    static CODEBOOKS: OnceLock<Codebooks> = OnceLock::new();
    CODEBOOKS.get_or_init(build)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::GetBits;

    #[test]
    fn the_tables_are_the_sizes_ffmpeg_reserves() {
        // the static VLCElem arrays of wmaprodec.c, and the shared buffer of the two coefficient
        // tables (2108 + 3912 entries)
        let books = codebooks();
        assert_eq!(books.scale.len(), 616);
        assert_eq!(books.scale_rl.len(), 1406);
        assert_eq!(books.vec4.len(), 604);
        assert_eq!(books.vec2.len(), 562);
        assert_eq!(books.vec1.len(), 562);
        assert_eq!(books.coef[0].len() + books.coef[1].len(), 2108 + 3912);
    }

    /// Each code of `table` (`entries` as FFmpeg lists them: length and symbol), followed by
    /// other bits, decodes to its symbol plus `offset` and reads its length.
    fn check(table: &[VlcElem], bits: u32, depth: u32, entries: &[(i8, i32)], offset: i32) {
        let mut code = 0u64;
        for &(length, symbol) in entries {
            let value = ((code >> (32 - length)) as u32) << (32 - length);
            let buffer = [
                (value >> 24) as u8,
                (value >> 16) as u8,
                (value >> 8) as u8,
                value as u8 | 0x5A >> (length as u32 % 8),
                0xC3,
                0x3C,
                0,
                0,
            ];
            let mut reader = GetBits::new(48);
            assert_eq!(reader.vlc(&buffer, table, bits, depth), symbol + offset);
            assert_eq!(reader.count(), i32::from(length));
            code += 1u64 << (32 - length);
        }
    }

    #[test]
    fn every_code_reads_back_its_symbol() {
        let books = codebooks();
        let vec4: Vec<(i8, i32)> = VEC4_LENS
            .iter()
            .zip(VEC4_SYMS)
            .map(|(&length, symbol)| (length as i8, i32::from(symbol)))
            .collect();
        check(&books.vec4, VLC_BITS, 2, &vec4, -1);
        let scale: Vec<(i8, i32)> = SCALE_TABLE
            .iter()
            .map(|&[symbol, length]| (length as i8, i32::from(symbol)))
            .collect();
        check(&books.scale, SCALE_VLC_BITS, 3, &scale, -60);
        let coef0: Vec<(i8, i32)> = COEF0_LENS
            .iter()
            .zip(COEF0_SYMS)
            .map(|(&length, symbol)| (length as i8, i32::from(symbol)))
            .collect();
        check(&books.coef[0], VLC_BITS, 3, &coef0, 0);
    }
}
