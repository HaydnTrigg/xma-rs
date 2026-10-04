// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavcodec/wmaprodec.c,
// libavcodec/wma.c, libavutil/float_dsp.c and libavutil/ffmath.h (FFmpeg n9.1-dev-56-gae4314e2f4);
// the original files carry these notices:
//
// Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
// Copyright (c) 2008 - 2011 Sascha Sommer, Benjamin Larsson
// Copyright (c) 2002-2007 The FFmpeg Project
// Copyright 2005 Balatoni Denes
// Copyright 2006 Loren Merritt
// Copyright (c) 2016 Ganesh Ajjanagadde <gajjanag@gmail.com>
// Copyright (c) 2026 the xma authors (the Rust port)
//
// xma is free software; you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation; either version
// 2.1 of the License, or (at your option) any later version.
//
// xma is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License (the LICENSE file) for more details.

//! FFmpeg's WMA Pro decoder (`libavcodec/wmaprodec.c`) as its XMA2 decoder runs it: one stream
//! of one or two channels, with the parameters `decode_init` derives for XMA2 (decode flags
//! `0x10d6`: frames of 512 samples with a 15-bit length prefix and DRC data, up to 4 subframes
//! of 512, 256 or 128 samples; 16 bits per sample; no channel mask, so no LFE channel).
//!
//! The control flow is FFmpeg's statement for statement, including how it recovers: a frame it
//! cannot decode is dropped where FFmpeg drops it, and only what FFmpeg logs at error level (or
//! returns as an error) is an error ([`ErrorLog`]). Names are FFmpeg's.

use crate::bits::{GetBits, PADDING, PutBits};
use crate::imdct::{self, Kernel};
use crate::tables::{
    COEF0_LEVEL, COEF0_RUN, COEF1_LEVEL, COEF1_RUN, CRITICAL_FREQ, QUANTIZERS, SCALE_RL_LEVEL,
    SCALE_RL_RUN, SINE_128, SINE_256, SINE_512,
};
use crate::vlc::{SCALE_VLC_BITS, VLC_BITS, VlcElem, codebooks};

const MAX_SUBFRAMES: usize = 32;
const MAX_BANDS: usize = 29;
const MAX_FRAMESIZE: usize = 32768;
const BLOCK_MAX_SIZE: usize = 8192;

/// `samples_per_frame`.
pub(crate) const SAMPLES_PER_FRAME: usize = 512;
/// `block_align`, which `xma_decode_init` forces: an XMA packet.
pub(crate) const BLOCK_ALIGN: usize = 2048;
/// `log2_frame_size`: `av_log2(block_align) + 4`.
const LOG2_FRAME_SIZE: u32 = 15;
/// `subframe_len_bits` (`max_subframe_len_bit` is set: 4 subframes at most).
const SUBFRAME_LEN_BITS: u32 = 2;
const MIN_SAMPLES_PER_SUBFRAME: usize = 128;
/// `num_possible_block_sizes`.
const BLOCK_SIZES: usize = 3;
/// `bits_per_sample`.
const BITS_PER_SAMPLE: i32 = 16;

/// `ff_exp10(exp / 20.0)` is `exp2(exp * this)` in FFmpeg's build: MSVC (`/fp:fast`) folds
/// `M_LOG2_10 * (exp / 20.0)` into one multiply by `M_LOG2_10 * 0.05`, rounded to this double.
/// Dividing by 20 instead rounds differently for some exponents, and the quantizer then differs
/// in its last bit.
pub(crate) const EXP10_FACTOR: f64 = f64::from_bits(0x3FC5_42A5_A12E_1C5B);

/// The first exponent [`QUANTIZERS`] holds.
const FIRST_QUANTIZER: i32 = -256;

/// The quantizer of exponent `exp` (`ff_exp10(exp / 20.0)`): held in [`QUANTIZERS`] for every
/// exponent real streams use, computed past them.
fn quantizer(exp: i32) -> f32 {
    exp.checked_sub(FIRST_QUANTIZER)
        .and_then(|index| usize::try_from(index).ok())
        .and_then(|index| QUANTIZERS.get(index).copied())
        .unwrap_or_else(|| (f64::from(exp) * EXP10_FACTOR).exp2() as f32)
}

/// The negative returns (`AVERROR_INVALIDDATA`, `AVERROR_PATCHWELCOME`, `AVERROR(EINVAL)`).
pub(crate) const INVALIDDATA: i32 = -1;
pub(crate) const PATCHWELCOME: i32 = -2;
pub(crate) const EINVAL: i32 = -3;

/// What FFmpeg logs at error level: the decoder goes on after it, the caller fails.
#[derive(Debug, Default)]
pub(crate) struct ErrorLog {
    pub count: u32,
    pub first: Option<String>,
}

impl ErrorLog {
    pub(crate) fn error(&mut self, message: String) {
        if self.first.is_none() {
            self.first = Some(message);
        }
        self.count += 1;
    }
}

/// `av_log2`.
pub(crate) fn av_log2(value: u32) -> u32 {
    if value == 0 {
        0
    } else {
        31 - value.leading_zeros()
    }
}

/// The frame a stream decodes into (`XMADecodeCtx.frames[stream]`): 512 samples per channel.
pub(crate) struct StreamFrame {
    pub samples: [Vec<f32>; 2],
    /// Whether it holds a buffer (`data[0]`): a skipped frame is unreferenced.
    pub allocated: bool,
}

impl StreamFrame {
    pub(crate) fn new() -> StreamFrame {
        StreamFrame {
            samples: [vec![0.0; SAMPLES_PER_FRAME], vec![0.0; SAMPLES_PER_FRAME]],
            allocated: false,
        }
    }
}

/// `WMAProChannelCtx`.
struct Channel {
    prev_block_len: usize,
    transmit_coefs: bool,
    num_subframes: usize,
    subframe_len: [u16; MAX_SUBFRAMES],
    subframe_offset: [u16; MAX_SUBFRAMES],
    cur_subframe: usize,
    decoded_samples: u16,
    grouped: bool,
    quant_step: i32,
    reuse_sf: bool,
    scale_factor_step: i32,
    max_scale_factor: i32,
    saved_scale_factors: [[i32; MAX_BANDS]; 2],
    scale_factor_idx: usize,
    /// Which of `saved_scale_factors` the subframe uses.
    scale_factors: usize,
    table_idx: usize,
    /// Where the subframe's coefficients are in `out`.
    coeffs: usize,
    num_vec_coeffs: usize,
    out: Vec<f32>,
}

impl Channel {
    fn new() -> Channel {
        Channel {
            prev_block_len: SAMPLES_PER_FRAME,
            transmit_coefs: false,
            num_subframes: 0,
            subframe_len: [0; MAX_SUBFRAMES],
            subframe_offset: [0; MAX_SUBFRAMES],
            cur_subframe: 0,
            decoded_samples: 0,
            grouped: false,
            quant_step: 0,
            reuse_sf: false,
            scale_factor_step: 0,
            max_scale_factor: 0,
            saved_scale_factors: [[0; MAX_BANDS]; 2],
            scale_factor_idx: 0,
            scale_factors: 0,
            table_idx: 0,
            coeffs: 0,
            num_vec_coeffs: 0,
            out: vec![0.0; BLOCK_MAX_SIZE + BLOCK_MAX_SIZE / 2],
        }
    }

    /// `subframe_len[index]`, where `index` may be one past the last subframe of 32: FFmpeg
    /// then reads the field after the array.
    fn subframe_len_at(&self, index: usize) -> u16 {
        self.subframe_len
            .get(index)
            .copied()
            .unwrap_or(self.subframe_offset[0])
    }
}

/// `WMAProChannelGrp` (a stream has two channels at most).
#[derive(Clone, Copy, Default)]
struct ChannelGroup {
    num_channels: usize,
    transform: bool,
    transform_band: [i8; MAX_BANDS],
    decorrelation_matrix: [f32; 4],
    channel_data: [usize; 2],
}

/// `WMAProDecodeCtx` of one XMA stream.
pub(crate) struct WmaPro {
    kernel: Kernel,
    nb_channels: usize,

    num_sfb: [usize; BLOCK_SIZES],
    sfb_offsets: [[i16; MAX_BANDS]; BLOCK_SIZES],
    sf_offsets: [[[i8; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES],

    /// The bit reservoir (`frame_data`), its writer and its reader.
    frame_data: Vec<u8>,
    pb: PutBits,
    gb: GetBits,

    next_packet_start: i32,
    packet_offset: u32,
    num_saved_bits: i32,
    frame_offset: i32,
    packet_loss: bool,
    pub packet_done: bool,
    pub eof_done: bool,

    frame_num: u32,
    buf_bit_size: i32,
    skip_frame: bool,
    parsed_all_subframes: bool,
    pub skip_packets: u8,
    pub trim_start: u16,
    pub trim_end: u16,

    subframe_len: usize,
    channels_for_cur_subframe: usize,
    channel_indexes_for_cur_subframe: [usize; 2],
    num_bands: usize,
    transmit_num_vec_coeffs: bool,
    table_idx: usize,

    num_chgroups: usize,
    chgroup: [ChannelGroup; 2],
    channel: Vec<Channel>,
    tmp: Vec<f32>,

    pub errors: ErrorLog,
}

/// `get_rate`: the rate whose band layout an XMA stream uses.
fn get_rate(sample_rate: i32) -> i32 {
    if sample_rate > 44100 {
        48000
    } else if sample_rate > 32000 {
        44100
    } else if sample_rate > 24000 {
        32000
    } else {
        24000
    }
}

/// The sine window of `len` (128, 256 or 512) samples.
fn sine_window(len: usize) -> &'static [f32] {
    match len {
        128 => &SINE_128,
        256 => &SINE_256,
        512 => &SINE_512,
        _ => panic!("a window of {len} samples"),
    }
}

impl WmaPro {
    /// `decode_init` for XMA2 stream config `nb_channels` (1 or 2) at `sample_rate`.
    pub(crate) fn new(
        kernel: Kernel,
        nb_channels: usize,
        sample_rate: i32,
    ) -> Result<WmaPro, String> {
        if !(1..=2).contains(&nb_channels) {
            return Err(format!(
                "invalid number of channels per XMA stream {nb_channels}"
            ));
        }
        let mut num_sfb = [0usize; BLOCK_SIZES];
        let mut sfb_offsets = [[0i16; MAX_BANDS]; BLOCK_SIZES];
        let rate = get_rate(sample_rate);
        for i in 0..BLOCK_SIZES {
            let subframe_len = (SAMPLES_PER_FRAME >> i) as i32;
            let mut band = 1usize;
            sfb_offsets[i][0] = 0;
            let mut x = 0;
            while x < MAX_BANDS - 1 && i32::from(sfb_offsets[i][band - 1]) < subframe_len {
                let mut offset = (subframe_len * 2 * i32::from(CRITICAL_FREQ[x])) / rate + 2;
                offset &= !3;
                if offset > i32::from(sfb_offsets[i][band - 1]) {
                    sfb_offsets[i][band] = offset as i16;
                    band += 1;
                }
                if offset >= subframe_len {
                    break;
                }
                x += 1;
            }
            sfb_offsets[i][band - 1] = subframe_len as i16;
            num_sfb[i] = band - 1;
            if num_sfb[i] == 0 {
                return Err("num_sfb invalid".to_owned());
            }
        }
        let mut sf_offsets = [[[0i8; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES];
        for i in 0..BLOCK_SIZES {
            for b in 0..num_sfb[i] {
                let offset =
                    ((i32::from(sfb_offsets[i][b]) + i32::from(sfb_offsets[i][b + 1]) - 1) << i)
                        >> 1;
                for x in 0..BLOCK_SIZES {
                    let mut v = 0usize;
                    while (i32::from(sfb_offsets[x][v + 1]) << x) < offset {
                        v += 1;
                        assert!(v < MAX_BANDS);
                    }
                    sf_offsets[i][x][b] = v as i8;
                }
            }
        }
        Ok(WmaPro {
            kernel,
            nb_channels,
            num_sfb,
            sfb_offsets,
            sf_offsets,
            frame_data: vec![0; MAX_FRAMESIZE + PADDING],
            pb: PutBits::default(),
            gb: GetBits::default(),
            next_packet_start: 0,
            packet_offset: 0,
            num_saved_bits: 0,
            frame_offset: 0,
            packet_loss: true,
            packet_done: false,
            eof_done: false,
            frame_num: 0,
            buf_bit_size: 0,
            skip_frame: true,
            parsed_all_subframes: false,
            skip_packets: 0,
            trim_start: 0,
            trim_end: 0,
            subframe_len: 0,
            channels_for_cur_subframe: 0,
            channel_indexes_for_cur_subframe: [0; 2],
            num_bands: 0,
            transmit_num_vec_coeffs: false,
            table_idx: 0,
            num_chgroups: 0,
            chgroup: [ChannelGroup::default(); 2],
            channel: (0..nb_channels).map(|_| Channel::new()).collect(),
            tmp: vec![0.0; BLOCK_MAX_SIZE],
            errors: ErrorLog::default(),
        })
    }

    #[inline(always)]
    fn bits(&mut self, count: u32) -> u32 {
        self.gb.get(&self.frame_data, count)
    }

    #[inline(always)]
    fn bit(&mut self) -> u32 {
        self.gb.bit(&self.frame_data)
    }

    #[inline(always)]
    fn vlc(&mut self, table: &[VlcElem], bits: u32, max_depth: u32) -> i32 {
        self.gb.vlc(&self.frame_data, table, bits, max_depth)
    }

    /// `decode_subframe_length`.
    fn decode_subframe_length(&mut self, offset: usize) -> i32 {
        if offset == SAMPLES_PER_FRAME - MIN_SAMPLES_PER_SUBFRAME {
            return MIN_SAMPLES_PER_SUBFRAME as i32;
        }
        if self.gb.left() < 1 {
            return INVALIDDATA;
        }
        let mut frame_len_shift = 0;
        if self.bit() != 0 {
            frame_len_shift = 1 + self.bits(SUBFRAME_LEN_BITS - 1);
        }
        let subframe_len = (SAMPLES_PER_FRAME >> frame_len_shift) as i32;
        if subframe_len < MIN_SAMPLES_PER_SUBFRAME as i32 || subframe_len > SAMPLES_PER_FRAME as i32
        {
            self.errors
                .error(format!("broken frame: subframe_len {subframe_len}"));
            return INVALIDDATA;
        }
        subframe_len
    }

    /// `decode_tilehdr`.
    fn decode_tilehdr(&mut self) -> i32 {
        let mut num_samples = [0usize; 2];
        let mut contains_subframe = [false; 2];
        let mut channels_for_cur_subframe = self.nb_channels;
        let mut min_channel_len = 0usize;
        for channel in &mut self.channel {
            channel.num_subframes = 0;
        }
        // max_num_subframes is 4, so the layout bit is there
        let fixed_channel_layout = self.bit() != 0;
        loop {
            for c in 0..self.nb_channels {
                contains_subframe[c] = if num_samples[c] == min_channel_len {
                    if fixed_channel_layout
                        || channels_for_cur_subframe == 1
                        || min_channel_len == SAMPLES_PER_FRAME - MIN_SAMPLES_PER_SUBFRAME
                    {
                        true
                    } else {
                        self.bit() != 0
                    }
                } else {
                    false
                };
            }
            let subframe_len = self.decode_subframe_length(min_channel_len);
            if subframe_len <= 0 {
                return INVALIDDATA;
            }
            let subframe_len = subframe_len as usize;
            min_channel_len += subframe_len;
            for c in 0..self.nb_channels {
                if contains_subframe[c] {
                    let channel = &mut self.channel[c];
                    if channel.num_subframes >= MAX_SUBFRAMES {
                        self.errors
                            .error("broken frame: num subframes > 31".to_owned());
                        return INVALIDDATA;
                    }
                    channel.subframe_len[channel.num_subframes] = subframe_len as u16;
                    num_samples[c] += subframe_len;
                    channel.num_subframes += 1;
                    if num_samples[c] > SAMPLES_PER_FRAME {
                        self.errors
                            .error("broken frame: channel len > samples_per_frame".to_owned());
                        return INVALIDDATA;
                    }
                } else if num_samples[c] <= min_channel_len {
                    if num_samples[c] < min_channel_len {
                        channels_for_cur_subframe = 0;
                        min_channel_len = num_samples[c];
                    }
                    channels_for_cur_subframe += 1;
                }
            }
            if min_channel_len >= SAMPLES_PER_FRAME {
                break;
            }
        }
        for channel in &mut self.channel {
            let mut offset = 0u16;
            for i in 0..channel.num_subframes {
                channel.subframe_offset[i] = offset;
                offset = offset.wrapping_add(channel.subframe_len[i]);
            }
        }
        0
    }

    /// `decode_channel_transform` (a stream's group holds two channels at most).
    fn decode_channel_transform(&mut self) -> i32 {
        self.num_chgroups = 0;
        if self.nb_channels > 1 {
            let mut remaining_channels = self.channels_for_cur_subframe;
            if self.bit() != 0 {
                // avpriv_request_sample("Channel transform bit"): a warning
                return PATCHWELCOME;
            }
            while remaining_channels > 0 && self.num_chgroups < self.channels_for_cur_subframe {
                let index = self.num_chgroups;
                let mut group = ChannelGroup {
                    num_channels: remaining_channels,
                    ..self.chgroup[index]
                };
                group.transform = false;
                let mut data = 0;
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    if !self.channel[c].grouped {
                        group.channel_data[data] = c;
                        data += 1;
                    }
                    self.channel[c].grouped = true;
                }
                if group.num_channels == 2 {
                    if self.bit() != 0 {
                        if self.bit() != 0 {
                            // avpriv_request_sample("Unknown channel transform type")
                            self.chgroup[index] = group;
                            return PATCHWELCOME;
                        }
                    } else {
                        group.transform = true;
                        // nb_channels is 2
                        group.decorrelation_matrix = [1.0, -1.0, 1.0, 1.0];
                    }
                }
                if group.transform {
                    if self.bit() == 0 {
                        for band in 0..self.num_bands {
                            group.transform_band[band] = self.bit() as i8;
                        }
                    } else {
                        group.transform_band[..self.num_bands].fill(1);
                    }
                }
                remaining_channels -= group.num_channels;
                self.chgroup[index] = group;
                self.num_chgroups += 1;
            }
        }
        0
    }

    /// `ff_wma_get_large_val`.
    fn get_large_val(&mut self) -> u32 {
        let mut n_bits = 8;
        if self.bit() != 0 {
            n_bits += 8;
            if self.bit() != 0 {
                n_bits += 8;
                if self.bit() != 0 {
                    n_bits += 7;
                }
            }
        }
        self.gb.get_long(&self.frame_data, n_bits)
    }

    /// `ff_wma_run_level_decode` as `decode_coeffs` calls it (version 1, the escape's offset in
    /// `esc_len` bits), into channel `c`'s coefficients from `offset` on.
    fn run_level_decode(&mut self, c: usize, table: usize, mut offset: usize) -> i32 {
        let books = codebooks();
        let vlc = &books.coef[table];
        let (run_table, level_table): (&[u16], &[f32]) = if table == 1 {
            (&COEF1_RUN, &COEF1_LEVEL)
        } else {
            (&COEF0_RUN, &COEF0_LEVEL)
        };
        let num_coefs = self.subframe_len;
        let coef_mask = self.subframe_len - 1;
        let frame_len_bits = av_log2(self.subframe_len as u32 - 1) + 1;
        let coeffs = self.channel[c].coeffs;
        while offset < num_coefs {
            let code = self.vlc(vlc, VLC_BITS, 3);
            if code > 1 {
                let code = code as usize;
                offset += usize::from(run_table[code]);
                let sign = self.bit().wrapping_sub(1);
                self.channel[c].out[coeffs + (offset & coef_mask)] =
                    f32::from_bits(level_table[code].to_bits() ^ (sign & 0x8000_0000));
            } else if code == 1 {
                break;
            } else {
                let level = self.get_large_val() as i32;
                if self.bit() != 0 {
                    if self.bit() != 0 {
                        if self.bit() != 0 {
                            self.errors.error("broken escape sequence".to_owned());
                            return INVALIDDATA;
                        }
                        offset += self.bits(frame_len_bits) as usize + 4;
                    } else {
                        offset += self.bits(2) as usize + 1;
                    }
                }
                let sign = self.bit() as i32 - 1;
                self.channel[c].out[coeffs + (offset & coef_mask)] =
                    ((level ^ sign).wrapping_sub(sign)) as f32;
            }
            offset += 1;
        }
        if offset > num_coefs {
            self.errors.error(format!(
                "overflow ({offset} > {num_coefs}) in spectral RLE, ignoring"
            ));
            return INVALIDDATA;
        }
        0
    }

    /// `decode_coeffs`.
    fn decode_coeffs(&mut self, c: usize) -> i32 {
        // the integers 0 to 15 as floats
        const FVAL: [u32; 16] = [
            0x0000_0000,
            0x3f80_0000,
            0x4000_0000,
            0x4040_0000,
            0x4080_0000,
            0x40a0_0000,
            0x40c0_0000,
            0x40e0_0000,
            0x4100_0000,
            0x4110_0000,
            0x4120_0000,
            0x4130_0000,
            0x4140_0000,
            0x4150_0000,
            0x4160_0000,
            0x4170_0000,
        ];
        let books = codebooks();
        let table = self.bit() as usize;
        let mut rl_mode = false;
        let mut cur_coeff = 0usize;
        let mut num_zeros = 0usize;
        let coeffs = self.channel[c].coeffs;
        let num_vec_coeffs = self.channel[c].num_vec_coeffs;
        while (self.transmit_num_vec_coeffs || !rl_mode) && cur_coeff + 3 < num_vec_coeffs {
            let mut vals = [0u32; 4];
            let idx = self.vlc(&books.vec4, VLC_BITS, 2);
            if idx < 0 {
                for i in [0, 2] {
                    let idx = self.vlc(&books.vec2, VLC_BITS, 2);
                    if idx < 0 {
                        let mut v0 = self.vlc(&books.vec1, VLC_BITS, 2) as u32;
                        if v0 == 100 {
                            v0 = v0.wrapping_add(self.get_large_val());
                        }
                        let mut v1 = self.vlc(&books.vec1, VLC_BITS, 2) as u32;
                        if v1 == 100 {
                            v1 = v1.wrapping_add(self.get_large_val());
                        }
                        vals[i] = (v0 as f32).to_bits();
                        vals[i + 1] = (v1 as f32).to_bits();
                    } else {
                        let idx = idx as usize;
                        vals[i] = FVAL[idx >> 4];
                        vals[i + 1] = FVAL[idx & 0xF];
                    }
                }
            } else {
                let idx = idx as usize;
                vals = [
                    FVAL[idx >> 12],
                    FVAL[(idx >> 8) & 0xF],
                    FVAL[(idx >> 4) & 0xF],
                    FVAL[idx & 0xF],
                ];
            }
            for value in vals {
                if value != 0 {
                    let sign = self.bit().wrapping_sub(1);
                    self.channel[c].out[coeffs + cur_coeff] = f32::from_bits(value ^ (sign << 31));
                    num_zeros = 0;
                } else {
                    self.channel[c].out[coeffs + cur_coeff] = 0.0;
                    num_zeros += 1;
                    rl_mode |= num_zeros > self.subframe_len >> 8;
                }
                cur_coeff += 1;
            }
        }
        if cur_coeff < self.subframe_len {
            self.channel[c].out[coeffs + cur_coeff..coeffs + self.subframe_len].fill(0.0);
            let result = self.run_level_decode(c, table, cur_coeff);
            if result < 0 {
                return result;
            }
        }
        0
    }

    /// `decode_scale_factors`.
    fn decode_scale_factors(&mut self) -> i32 {
        let books = codebooks();
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let num_bands = self.num_bands;
            let used = 1 - self.channel[c].scale_factor_idx;
            self.channel[c].scale_factors = used;
            if self.channel[c].reuse_sf {
                let channel = &mut self.channel[c];
                let sf_offsets = &self.sf_offsets[self.table_idx][channel.table_idx];
                let saved = channel.saved_scale_factors[channel.scale_factor_idx];
                for b in 0..num_bands {
                    channel.saved_scale_factors[used][b] = saved[sf_offsets[b] as usize];
                }
            }
            if self.channel[c].cur_subframe == 0 || self.bit() != 0 {
                if !self.channel[c].reuse_sf {
                    let step = self.bits(2) as i32 + 1;
                    self.channel[c].scale_factor_step = step;
                    let mut val = 45 / step;
                    for b in 0..num_bands {
                        val += self.vlc(&books.scale, SCALE_VLC_BITS, 3);
                        self.channel[c].saved_scale_factors[used][b] = val;
                    }
                } else {
                    let mut b = 0usize;
                    while b < num_bands {
                        let idx = self.vlc(&books.scale_rl, VLC_BITS, 3);
                        let (skip, val, sign);
                        if idx == 0 {
                            let code = self.bits(14);
                            val = (code >> 6) as i32;
                            sign = (code & 1) as i32 - 1;
                            skip = ((code & 0x3f) >> 1) as usize;
                        } else if idx == 1 {
                            break;
                        } else if idx < 0 {
                            // FFmpeg indexes its run table with -1 here
                            self.errors.error("invalid scale factor code".to_owned());
                            return INVALIDDATA;
                        } else {
                            skip = usize::from(SCALE_RL_RUN[idx as usize]);
                            val = i32::from(SCALE_RL_LEVEL[idx as usize]);
                            sign = self.bit() as i32 - 1;
                        }
                        b += skip;
                        if b >= num_bands {
                            self.errors.error("invalid scale factor coding".to_owned());
                            return INVALIDDATA;
                        }
                        self.channel[c].saved_scale_factors[used][b] += (val ^ sign) - sign;
                        b += 1;
                    }
                }
                let channel = &mut self.channel[c];
                channel.scale_factor_idx = 1 - channel.scale_factor_idx;
                channel.table_idx = self.table_idx;
                channel.reuse_sf = true;
            }
            let channel = &mut self.channel[c];
            let sf = &channel.saved_scale_factors[used];
            channel.max_scale_factor = sf[1..num_bands].iter().fold(sf[0], |max, &v| max.max(v));
        }
        0
    }

    /// `inverse_channel_transform`.
    fn inverse_channel_transform(&mut self) {
        let offsets = self.sfb_offsets[self.table_idx];
        for group in &self.chgroup[..self.num_chgroups] {
            if !group.transform {
                continue;
            }
            // a transform is decoded for two channels only
            let [first, second] = group.channel_data;
            let (one, two) = if first < second {
                let (low, high) = self.channel.split_at_mut(second);
                (&mut low[first], &mut high[0])
            } else {
                let (low, high) = self.channel.split_at_mut(first);
                (&mut high[0], &mut low[second])
            };
            let (one_at, two_at) = (one.coeffs, two.coeffs);
            for band in 0..self.num_bands {
                let start = offsets[band] as usize;
                let end = (offsets[band + 1] as usize).min(self.subframe_len);
                if group.transform_band[band] == 1 {
                    let mat = &group.decorrelation_matrix;
                    for y in start..end {
                        let data = [one.out[one_at + y], two.out[two_at + y]];
                        let mut sum = 0.0f32;
                        sum += data[0] * mat[0];
                        sum += data[1] * mat[1];
                        one.out[one_at + y] = sum;
                        let mut sum = 0.0f32;
                        sum += data[0] * mat[2];
                        sum += data[1] * mat[3];
                        two.out[two_at + y] = sum;
                    }
                } else if self.nb_channels == 2 {
                    // vector_fmul_scalar by 181.0 / 128
                    let factor = (181.0f64 / 128.0) as f32;
                    for value in &mut one.out[one_at + start..one_at + end] {
                        *value *= factor;
                    }
                    for value in &mut two.out[two_at + start..two_at + end] {
                        *value *= factor;
                    }
                }
            }
        }
    }

    /// `wmapro_window`: the sine window and the overlap-add with the previous subframe.
    fn window(&mut self) {
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let channel = &mut self.channel[c];
            let mut winlen = channel.prev_block_len;
            let mut start = channel.coeffs - (winlen >> 1);
            if self.subframe_len < winlen {
                start += (winlen - self.subframe_len) >> 1;
                winlen = self.subframe_len;
            }
            let window = sine_window(winlen);
            let len = winlen >> 1;
            // vector_fmul_window(start, start, start + len, window, len), in place
            let samples = &mut channel.out[start..start + winlen];
            for k in 0..len {
                let j = 2 * len - 1 - k;
                let (s0, s1) = (samples[k], samples[j]);
                let (wi, wj) = (window[k], window[j]);
                samples[k] = s0 * wj - s1 * wi;
                samples[j] = s0 * wi + s1 * wj;
            }
            channel.prev_block_len = self.subframe_len;
        }
    }

    /// `decode_subframe`.
    fn decode_subframe(&mut self) -> i32 {
        let mut offset = SAMPLES_PER_FRAME;
        let mut subframe_len = SAMPLES_PER_FRAME;
        let mut total_samples = (SAMPLES_PER_FRAME * self.nb_channels) as i32;
        let mut transmit_coeffs = false;

        for channel in &mut self.channel {
            channel.grouped = false;
            if offset > usize::from(channel.decoded_samples) {
                offset = usize::from(channel.decoded_samples);
                subframe_len = usize::from(channel.subframe_len_at(channel.cur_subframe));
            }
        }

        self.channels_for_cur_subframe = 0;
        for (i, channel) in self.channel.iter_mut().enumerate() {
            let cur_subframe = channel.cur_subframe;
            total_samples -= i32::from(channel.decoded_samples);
            if offset == usize::from(channel.decoded_samples)
                && subframe_len == usize::from(channel.subframe_len_at(cur_subframe))
            {
                total_samples -= i32::from(channel.subframe_len_at(cur_subframe));
                channel.decoded_samples = channel
                    .decoded_samples
                    .wrapping_add(channel.subframe_len_at(cur_subframe));
                self.channel_indexes_for_cur_subframe[self.channels_for_cur_subframe] = i;
                self.channels_for_cur_subframe += 1;
            }
        }
        if total_samples == 0 {
            self.parsed_all_subframes = true;
        }
        if subframe_len == 0 {
            // a subframe length FFmpeg would divide by
            self.errors.error("broken subframe length".to_owned());
            return INVALIDDATA;
        }

        self.table_idx = av_log2((SAMPLES_PER_FRAME / subframe_len) as u32) as usize;
        if self.table_idx >= BLOCK_SIZES {
            self.errors.error("broken subframe length".to_owned());
            return INVALIDDATA;
        }
        self.num_bands = self.num_sfb[self.table_idx];

        offset += SAMPLES_PER_FRAME >> 1;
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            self.channel[c].coeffs = offset;
        }

        self.subframe_len = subframe_len;

        // skip the extended header
        if self.bit() != 0 {
            let mut num_fill_bits = self.bits(2) as i32;
            if num_fill_bits == 0 {
                let len = self.bits(4);
                num_fill_bits = self.gb.get_z(&self.frame_data, len) as i32 + 1;
            }
            if num_fill_bits >= 0 {
                if self.gb.count() + num_fill_bits > self.num_saved_bits {
                    self.errors.error("invalid number of fill bits".to_owned());
                    return INVALIDDATA;
                }
                self.gb.skip_long(num_fill_bits);
            }
        }

        if self.bit() != 0 {
            // avpriv_request_sample("Reserved bit"): a warning
            return PATCHWELCOME;
        }

        if self.decode_channel_transform() < 0 {
            return INVALIDDATA;
        }

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let transmit = self.bit() != 0;
            self.channel[c].transmit_coefs = transmit;
            transmit_coeffs |= transmit;
        }

        if transmit_coeffs {
            let mut quant_step = (90 * BITS_PER_SAMPLE) >> 4;
            self.transmit_num_vec_coeffs = self.bit() != 0;
            if self.transmit_num_vec_coeffs {
                let num_bits = av_log2(self.subframe_len.div_ceil(4) as u32) + 1;
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    let num_vec_coeffs = (self.bits(num_bits) << 2) as usize;
                    if num_vec_coeffs > self.subframe_len {
                        self.errors
                            .error(format!("num_vec_coeffs {num_vec_coeffs} is too large"));
                        return INVALIDDATA;
                    }
                    self.channel[c].num_vec_coeffs = num_vec_coeffs;
                }
            } else {
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].num_vec_coeffs = self.subframe_len;
                }
            }
            // the quantization step
            let mut step = self.gb.get_signed(&self.frame_data, 6);
            quant_step += step;
            if step == -32 || step == 31 {
                let sign = i32::from(step == 31) - 1;
                let mut quant = 0;
                while self.gb.count() + 5 < self.num_saved_bits {
                    step = self.bits(5) as i32;
                    if step != 31 {
                        break;
                    }
                    quant += 31;
                }
                quant_step += ((quant + step) ^ sign) - sign;
            }
            // and its modifier per channel
            if self.channels_for_cur_subframe == 1 {
                let c = self.channel_indexes_for_cur_subframe[0];
                self.channel[c].quant_step = quant_step;
            } else {
                let modifier_len = self.bits(3);
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].quant_step = quant_step;
                    if self.bit() != 0 {
                        if modifier_len != 0 {
                            let modifier = self.bits(modifier_len) as i32 + 1;
                            self.channel[c].quant_step += modifier;
                        } else {
                            self.channel[c].quant_step += 1;
                        }
                    }
                }
            }
            if self.decode_scale_factors() < 0 {
                return INVALIDDATA;
            }
        }

        // the coefficients
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            if self.channel[c].transmit_coefs && self.gb.count() < self.num_saved_bits {
                // FFmpeg goes on whatever it returns
                self.decode_coeffs(c);
            } else {
                let coeffs = self.channel[c].coeffs;
                self.channel[c].out[coeffs..coeffs + subframe_len].fill(0.0);
            }
        }

        if transmit_coeffs {
            self.inverse_channel_transform();
            let offsets = self.sfb_offsets[self.table_idx];
            for i in 0..self.channels_for_cur_subframe {
                let c = self.channel_indexes_for_cur_subframe[i];
                let channel = &mut self.channel[c];
                let sf = &channel.saved_scale_factors[channel.scale_factors];
                let coeffs = channel.coeffs;
                // inverse quantization and rescaling
                for b in 0..self.num_bands {
                    let end = (offsets[b + 1] as usize).min(self.subframe_len);
                    let exp = channel.quant_step
                        - (channel.max_scale_factor - sf[b]) * channel.scale_factor_step;
                    let quant = quantizer(exp);
                    let start = offsets[b] as usize;
                    for k in start..end {
                        self.tmp[k] = channel.out[coeffs + k] * quant;
                    }
                }
                imdct::inverse(
                    self.kernel,
                    subframe_len,
                    &mut channel.out[coeffs..coeffs + subframe_len],
                    &self.tmp[..subframe_len],
                );
            }
        }

        self.window();

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            if self.channel[c].cur_subframe >= self.channel[c].num_subframes {
                self.errors.error("broken subframe".to_owned());
                return INVALIDDATA;
            }
            self.channel[c].cur_subframe += 1;
        }
        0
    }

    /// `decode_frame`: one frame from the reservoir into `frame`. Returns whether more frames
    /// follow in the packet (the trailer bit).
    fn decode_frame(&mut self, frame: &mut StreamFrame, got_frame: &mut bool) -> bool {
        let len = self.bits(LOG2_FRAME_SIZE) as i32;

        if self.decode_tilehdr() != 0 {
            self.packet_loss = true;
            return false;
        }

        // the postproc transform
        if self.nb_channels > 1 && self.bit() != 0 && self.bit() != 0 {
            for _ in 0..self.nb_channels * self.nb_channels {
                self.gb.skip(4);
            }
        }

        // the DRC gain
        let _drc_gain = self.bits(8);

        if self.bit() != 0 {
            if self.bit() != 0 {
                self.trim_start = self.bits(av_log2((SAMPLES_PER_FRAME * 2) as u32)) as u16;
            }
            if self.bit() != 0 {
                self.trim_end = self.bits(av_log2((SAMPLES_PER_FRAME * 2) as u32)) as u16;
            }
        } else {
            self.trim_start = 0;
            self.trim_end = 0;
        }

        self.parsed_all_subframes = false;
        for channel in &mut self.channel {
            channel.decoded_samples = 0;
            channel.cur_subframe = 0;
            channel.reuse_sf = false;
        }

        while !self.parsed_all_subframes {
            if self.decode_subframe() < 0 {
                self.packet_loss = true;
                return false;
            }
        }

        for (c, channel) in self.channel.iter_mut().enumerate() {
            frame.samples[c].copy_from_slice(&channel.out[..SAMPLES_PER_FRAME]);
            // the second half of the IMDCT output is the next frame's start
            channel.out.copy_within(
                SAMPLES_PER_FRAME..SAMPLES_PER_FRAME + SAMPLES_PER_FRAME / 2,
                0,
            );
        }

        if self.skip_frame {
            self.skip_frame = false;
            *got_frame = false;
            frame.allocated = false;
        } else {
            *got_frame = true;
        }

        let consumed = self.gb.count() - self.frame_offset;
        if len != consumed + 2 {
            self.errors.error(format!(
                "frame[{}] would have to skip {} bits",
                self.frame_num,
                len - consumed - 1
            ));
            self.packet_loss = true;
            return false;
        }
        self.gb.skip_long(len - consumed - 1);

        let more_frames = self.bit() != 0;
        self.frame_num += 1;
        more_frames
    }

    /// `save_bits`: `len` bits of the packet `data` into the reservoir, appended to what it
    /// holds or replacing it.
    fn save_bits(&mut self, gb: &mut GetBits, data: &[u8], len: i32, append: bool) {
        let buflen = if !append {
            self.frame_offset = gb.count() & 7;
            self.num_saved_bits = self.frame_offset;
            self.pb.reset();
            (self.num_saved_bits + len + 7) >> 3
        } else {
            (self.pb.count() as i32 + len + 7) >> 3
        };
        if len <= 0 || buflen > MAX_FRAMESIZE as i32 {
            // avpriv_request_sample("Too small input buffer"): a warning
            self.packet_loss = true;
            return;
        }
        self.num_saved_bits += len;
        let mut len = len;
        if !append {
            let start = gb.byte();
            self.pb.copy(
                &mut self.frame_data,
                &data[start..],
                self.num_saved_bits as usize,
            );
        } else {
            let align = (8 - (gb.count() & 7)).min(len);
            let value = gb.get(data, align as u32);
            self.pb.put(&mut self.frame_data, align as u32, value);
            len -= align;
            let start = gb.byte();
            self.pb
                .copy(&mut self.frame_data, &data[start..], len as usize);
        }
        gb.skip_long(len);
        self.gb = GetBits::new(self.num_saved_bits as u32);
        self.gb.skip(self.frame_offset as u32);
    }

    /// `decode_packet` for XMA2: one call on the `size` bytes of the packet left from `data`
    /// (which goes on past them: the next packets, then the padding). Returns the bytes it
    /// consumed, or a negative error; `got_frame` says whether it decoded a frame into `frame`.
    /// `size` 0 is the end of the stream: the samples held back come out.
    pub(crate) fn decode_packet(
        &mut self,
        data: &[u8],
        size: usize,
        frame: &mut StreamFrame,
        got_frame: &mut bool,
    ) -> i32 {
        *got_frame = false;
        let mut gb;

        if size == 0 {
            self.packet_done = false;
            if self.eof_done {
                return 0;
            }
            for (c, channel) in self.channel.iter().enumerate() {
                frame.samples[c].fill(0.0);
                frame.samples[c][..SAMPLES_PER_FRAME / 2]
                    .copy_from_slice(&channel.out[..SAMPLES_PER_FRAME / 2]);
            }
            self.eof_done = true;
            self.packet_done = true;
            *got_frame = true;
            return 0;
        } else if self.packet_done || self.packet_loss {
            self.packet_done = false;
            let buf_size = size.min(BLOCK_ALIGN);
            self.next_packet_start = (size - buf_size) as i32;
            self.buf_bit_size = (buf_size << 3) as i32;
            gb = GetBits::new((buf_size << 3) as u32);

            // the packet header: frames, the bits of the frame the last packet began, skip
            let _num_frames = gb.get(data, 6);
            let mut num_bits_prev_frame = gb.get(data, LOG2_FRAME_SIZE) as i32;
            gb.skip(3);
            self.skip_packets = gb.get(data, 8) as u8;

            if num_bits_prev_frame > 0 {
                let remaining_packet_bits = self.buf_bit_size - gb.count();
                if num_bits_prev_frame >= remaining_packet_bits {
                    num_bits_prev_frame = remaining_packet_bits;
                    self.packet_done = true;
                }
                // complete the frame the last packet began, and decode it
                self.save_bits(&mut gb, data, num_bits_prev_frame, true);
                if !self.packet_loss {
                    self.decode_frame(frame, got_frame);
                }
            }

            if self.packet_loss {
                // not an incomplete frame to go on with
                self.num_saved_bits = 0;
                self.packet_loss = false;
            }
        } else {
            if (size as i32) < self.next_packet_start {
                self.packet_loss = true;
                return INVALIDDATA;
            }
            let bytes = size as i32 - self.next_packet_start;
            self.buf_bit_size = bytes << 3;
            gb = GetBits::new((bytes << 3) as u32);
            gb.skip(self.packet_offset);
            let remaining = self.buf_bit_size - gb.count();
            let frame_size = if remaining > LOG2_FRAME_SIZE as i32 {
                gb.show(data, LOG2_FRAME_SIZE) as i32
            } else {
                0
            };
            if frame_size != 0 && frame_size <= remaining {
                self.save_bits(&mut gb, data, frame_size, false);
                if !self.packet_loss {
                    self.packet_done = !self.decode_frame(frame, got_frame);
                }
            } else {
                self.packet_done = true;
            }
        }

        let remaining = self.buf_bit_size - gb.count();
        if remaining < 0 {
            self.errors.error(format!("Overread {}", -remaining));
            self.packet_loss = true;
        }

        if self.packet_done && !self.packet_loss && remaining > 0 {
            // the start of the frame the next packet completes
            self.save_bits(&mut gb, data, remaining, false);
        }

        self.packet_offset = (gb.count() & 7) as u32;
        if self.packet_loss {
            return INVALIDDATA;
        }
        gb.count() >> 3
    }

    /// `packet_loss`: the packet just decoded was not decodable.
    pub(crate) fn packet_loss(&self) -> bool {
        self.packet_loss
    }
}
