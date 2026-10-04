// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavcodec/get_bits.h,
// libavcodec/put_bits.h and libavcodec/bitstream.c (FFmpeg n9.1-dev-56-gae4314e2f4); the original
// files carry these notices:
//
// Copyright (c) 2004 Michael Niedermayer <michaelni@gmx.at>
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

//! FFmpeg's bit reader (`libavcodec/get_bits.h`: big endian, the checked reader FFmpeg builds by
//! default) and the bit copy its WMA Pro decoder fills the frame reservoir with
//! (`libavcodec/put_bits.h`, `ff_copy_bits`).
//!
//! The reader keeps no reference to its buffer: every read is given the buffer, so a decoder
//! can hold a reader over a buffer of its own and still be borrowed mutably. What a read past
//! the end returns is whatever the buffer holds there, as in FFmpeg: the index stops 8 bits past
//! the end (`size_in_bits_plus8`), and every buffer this crate reads carries padding after its
//! bits, so a valid stream never depends on those bytes.

/// The bytes of padding every buffer read here carries past its bits
/// (`AV_INPUT_BUFFER_PADDING_SIZE`).
pub(crate) const PADDING: usize = 64;

/// `GetBitContext` without its buffer pointer.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GetBits {
    index: u32,
    size_in_bits: u32,
    size_plus8: u32,
}

/// `AV_RB32` at `at`; bytes past the slice read as zero (never met with padded buffers).
#[inline(always)]
fn rb32(buffer: &[u8], at: usize) -> u32 {
    match buffer.get(at..at + 4) {
        Some(bytes) => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        None => {
            let mut bytes = [0u8; 4];
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = buffer.get(at + index).copied().unwrap_or(0);
            }
            u32::from_be_bytes(bytes)
        }
    }
}

/// `AV_RB64` at `at`, as [`rb32`].
#[inline(always)]
fn rb64(buffer: &[u8], at: usize) -> u64 {
    match buffer.get(at..at + 8) {
        Some(bytes) => u64::from_be_bytes(bytes.try_into().expect("8 bytes")),
        None => {
            let mut bytes = [0u8; 8];
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = buffer.get(at + index).copied().unwrap_or(0);
            }
            u64::from_be_bytes(bytes)
        }
    }
}

impl GetBits {
    /// `init_get_bits`: a reader of `size_in_bits` bits.
    pub(crate) fn new(size_in_bits: u32) -> GetBits {
        GetBits {
            index: 0,
            size_in_bits,
            size_plus8: size_in_bits + 8,
        }
    }

    /// `get_bits_count`.
    #[inline(always)]
    pub(crate) fn count(&self) -> i32 {
        self.index as i32
    }

    /// `get_bits_left`.
    #[inline(always)]
    pub(crate) fn left(&self) -> i32 {
        self.size_in_bits as i32 - self.index as i32
    }

    /// The 32 bits from the index on (`UPDATE_CACHE`).
    #[inline(always)]
    fn cache(&self, buffer: &[u8], index: u32) -> u32 {
        rb32(buffer, (index >> 3) as usize) << (index & 7)
    }

    /// `SKIP_COUNTER` of the checked reader.
    #[inline(always)]
    fn skip_counter(&self, index: u32, count: u32) -> u32 {
        self.size_plus8.min(index.wrapping_add(count))
    }

    /// `get_bits`, 1 to 25 bits.
    #[inline(always)]
    pub(crate) fn get(&mut self, buffer: &[u8], count: u32) -> u32 {
        debug_assert!((1..=25).contains(&count));
        let value = self.cache(buffer, self.index) >> (32 - count);
        self.index = self.skip_counter(self.index, count);
        value
    }

    /// `get_bitsz`: 0 to 25 bits.
    #[inline(always)]
    pub(crate) fn get_z(&mut self, buffer: &[u8], count: u32) -> u32 {
        if count == 0 {
            0
        } else {
            self.get(buffer, count)
        }
    }

    /// `get_bits_long`, 0 to 32 bits (the 64-bit cache of an x86-64 build).
    #[inline(always)]
    pub(crate) fn get_long(&mut self, buffer: &[u8], count: u32) -> u32 {
        if count == 0 {
            return 0;
        }
        let cache = ((rb64(buffer, (self.index >> 3) as usize) << (self.index & 7)) >> 32) as u32;
        let value = cache >> (32 - count);
        self.index = self.skip_counter(self.index, count);
        value
    }

    /// `get_sbits`, 1 to 25 bits.
    #[inline(always)]
    pub(crate) fn get_signed(&mut self, buffer: &[u8], count: u32) -> i32 {
        let value = (self.cache(buffer, self.index) as i32) >> (32 - count);
        self.index = self.skip_counter(self.index, count);
        value
    }

    /// `show_bits`, 1 to 25 bits.
    #[inline(always)]
    pub(crate) fn show(&self, buffer: &[u8], count: u32) -> u32 {
        self.cache(buffer, self.index) >> (32 - count)
    }

    /// `get_bits1`.
    #[inline(always)]
    pub(crate) fn bit(&mut self, buffer: &[u8]) -> u32 {
        let byte = buffer.get((self.index >> 3) as usize).copied().unwrap_or(0);
        let value = u32::from((byte << (self.index & 7)) >> 7);
        if self.index < self.size_plus8 {
            self.index += 1;
        }
        value
    }

    /// `skip_bits`.
    #[inline(always)]
    pub(crate) fn skip(&mut self, count: u32) {
        self.index = self.skip_counter(self.index, count);
    }

    /// `skip_bits_long`: `count` clipped to the start and to 8 bits past the end.
    pub(crate) fn skip_long(&mut self, count: i32) {
        let index = self.index as i32;
        self.index = (index + count.clamp(-index, self.size_plus8 as i32 - index)) as u32;
    }

    /// `get_vlc2`: a code of `table` (`bits` bits at its root, at most `max_depth` levels).
    /// What a code the table does not hold reads is FFmpeg's: the symbol -1 and the bits up to
    /// the level that has no entry.
    #[inline(always)]
    pub(crate) fn vlc(
        &mut self,
        buffer: &[u8],
        table: &[crate::vlc::VlcElem],
        bits: u32,
        max_depth: u32,
    ) -> i32 {
        let mut index = self.index;
        let entry = table[(self.cache(buffer, index) >> (32 - bits)) as usize];
        let mut code = i32::from(entry.sym);
        let mut length = i32::from(entry.len);
        if max_depth > 1 && length < 0 {
            index = self.skip_counter(index, bits);
            let bits = (-length) as u32;
            let at = (self.cache(buffer, index) >> (32 - bits)) as i32 + code;
            let entry = table[at as usize];
            code = i32::from(entry.sym);
            length = i32::from(entry.len);
            if max_depth > 2 && length < 0 {
                index = self.skip_counter(index, bits);
                let bits = (-length) as u32;
                let at = (self.cache(buffer, index) >> (32 - bits)) as i32 + code;
                let entry = table[at as usize];
                code = i32::from(entry.sym);
                length = i32::from(entry.len);
            }
        }
        self.index = self.skip_counter(index, length as u32);
        code
    }

    /// The byte the index is in (`gb->buffer + (get_bits_count(gb) >> 3)`).
    #[inline(always)]
    pub(crate) fn byte(&self) -> usize {
        (self.index >> 3) as usize
    }
}

/// `PutBitContext` over a buffer it does not own: the bits written so far. Every write leaves
/// the buffer as FFmpeg's `flush_put_bits` on a copy of the context would (the bits, the rest
/// of their last byte zero, nothing after it touched): the frame reservoir is read in that
/// state.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PutBits {
    bits: usize,
}

impl PutBits {
    /// `put_bits_count`.
    pub(crate) fn count(&self) -> usize {
        self.bits
    }

    /// `init_put_bits`.
    pub(crate) fn reset(&mut self) {
        self.bits = 0;
    }

    /// `put_bits`: the low `count` bits of `value` (1 to 8).
    pub(crate) fn put(&mut self, buffer: &mut [u8], count: u32, value: u32) {
        debug_assert!((1..=8).contains(&count));
        let byte = ((value << (8 - count)) & 0xFF) as u8;
        self.copy(buffer, &[byte], count as usize);
    }

    /// `ff_copy_bits`: the first `length` bits of `source`.
    pub(crate) fn copy(&mut self, buffer: &mut [u8], source: &[u8], length: usize) {
        if length == 0 {
            return;
        }
        let shift = self.bits & 7;
        let mut at = self.bits >> 3;
        let whole = length >> 3;
        let rest = length & 7;
        // the last source byte with only its `rest` bits
        let last = if rest > 0 {
            source[whole] & (0xFFu8 << (8 - rest))
        } else {
            0
        };
        if shift == 0 {
            buffer[at..at + whole].copy_from_slice(&source[..whole]);
            if rest > 0 {
                buffer[at + whole] = last;
            }
        } else {
            // the bits already in the first byte stay
            let mut carry = buffer[at] & (0xFFu8 << (8 - shift));
            for &byte in &source[..whole] {
                buffer[at] = carry | (byte >> shift);
                carry = byte << (8 - shift);
                at += 1;
            }
            if rest > 0 {
                buffer[at] = carry | (last >> shift);
                if shift + rest > 8 {
                    buffer[at + 1] = last << (8 - shift);
                }
            } else {
                buffer[at] = carry;
            }
        }
        self.bits += length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reader_reads_big_endian_and_stops_past_the_end() {
        let buffer = [0b1011_0011, 0b0101_0101, 0xFF, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let mut bits = GetBits::new(12);
        assert_eq!(bits.get(&buffer, 3), 0b101);
        assert_eq!(bits.bit(&buffer), 1);
        assert_eq!(bits.show(&buffer, 4), 0b0011);
        assert_eq!(bits.get_signed(&buffer, 4), 3);
        bits.skip(1);
        assert_eq!(bits.get_signed(&buffer, 3), -3);
        assert_eq!(bits.count(), 12);
        assert_eq!(bits.left(), 0);
        // past the end the reader stops 8 bits on, and reads what the buffer holds
        assert_eq!(bits.get_long(&buffer, 32), 0x5FF0_0000);
        assert_eq!(bits.count(), 20);
        bits.skip_long(-100);
        assert_eq!(bits.count(), 0);
    }

    #[test]
    fn the_writer_copies_bits_at_any_offset() {
        let source = [0xAB, 0xCD, 0xEF];
        let mut buffer = [0x55u8; 8];
        let mut put = PutBits::default();
        put.put(&mut buffer, 3, 0b110);
        put.copy(&mut buffer, &source, 13);
        assert_eq!(put.count(), 16);
        // 110 then 1010 1011 1100 1
        assert_eq!(buffer[0], 0b1101_0101);
        assert_eq!(buffer[1], 0b0111_1001);
        // and the bytes after the bits are left alone
        assert_eq!(buffer[2], 0x55);
        put.copy(&mut buffer, &source, 5);
        assert_eq!(buffer[2], 0b1010_1000);
    }
}
