// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavcodec/wmaprodec.c and
// libavcodec/decode.c (FFmpeg n9.1-dev-56-gae4314e2f4); the original files carry these notices:
//
// Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
// Copyright (c) 2008 - 2011 Sascha Sommer, Benjamin Larsson
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

//! FFmpeg's XMA2 decoder around the WMA Pro one (`xma_decode_packet` in `wmaprodec.c`) for a
//! single stream, and what of libavcodec's decode loop (`decode_simple_internal`,
//! `discard_samples`) the samples depend on, with the stream sent as one packet and the decoder
//! drained after it.
//!
//! What that adds to the frames the stream decodes: the first frame is never output (the WMA Pro
//! decoder skips it), the decoded samples go through a FIFO that holds the last 4096 back until
//! the end, the first 64 samples out are discarded (`skip_samples`), and at the end the stream
//! gives the samples its overlap still holds and loses `trim_start + trim_end - 192` of its
//! last ones. A stream is invalid when FFmpeg would have returned an error or logged one at
//! error level ([`crate::Error::Invalid`]).

use crate::Error;
use crate::bits::PADDING;
use crate::imdct::Kernel;
use crate::wmapro::{BLOCK_ALIGN, EINVAL, ErrorLog, SAMPLES_PER_FRAME, StreamFrame, WmaPro};

/// The samples the FIFO keeps back while packets come (`nb_samples -= FFMIN(nb_samples, 4096)`).
const HELD_BACK: usize = 4096;
/// `skip_samples` of a stream's first frame.
const SKIP_SAMPLES: i32 = 64;
/// How many calls in a row may consume nothing before the stream is called stuck (FFmpeg would
/// loop for ever; a valid stream never does it twice).
const STALL_LIMIT: u32 = 64;

/// One stream's decoder: `XMADecodeCtx` with one stream, and the codec context's
/// `skip_samples`.
struct Xma {
    channels: usize,
    stream: WmaPro,
    frame: StreamFrame,
    /// `samples[channel][0]`: what was decoded and not yet read, from `fifo_start` on.
    fifo: [Vec<f32>; 2],
    fifo_start: usize,
    trim_start: i32,
    trim_end: i32,
    flushed: bool,
    skip_samples: i32,
    errors: ErrorLog,
    /// The samples out, interleaved.
    out: Vec<f32>,
}

impl Xma {
    fn fifo_len(&self) -> usize {
        self.fifo[0].len() - self.fifo_start
    }

    /// A frame of `count` samples read from the FIFO, through `discard_samples`.
    fn output(&mut self, count: usize) {
        let start = self.fifo_start;
        self.fifo_start += count;
        let mut skip = 0;
        if self.skip_samples > 0 {
            if count as i32 <= self.skip_samples {
                self.skip_samples -= count as i32;
                return;
            }
            skip = self.skip_samples as usize;
            self.skip_samples = 0;
        }
        self.out.reserve((count - skip) * self.channels);
        for index in start + skip..start + count {
            for channel in 0..self.channels {
                self.out.push(self.fifo[channel][index]);
            }
        }
        // what was read is gone
        if self.fifo_start >= 1 << 16 {
            for channel in &mut self.fifo[..self.channels] {
                channel.drain(..self.fifo_start);
            }
            self.fifo_start = 0;
        }
    }

    /// `xma_decode_packet` on the `size` bytes of the packet left from `data` (0 at the end of
    /// the stream). Returns the bytes consumed or a negative error, and whether a frame came
    /// out.
    fn decode(&mut self, data: &[u8], size: usize) -> (i32, bool) {
        if !self.frame.allocated {
            self.skip_samples = SKIP_SAMPLES;
            self.frame.allocated = true;
        }
        let mut got_stream_frame = false;
        let mut result = 0;
        if !self.stream.eof_done {
            result = self
                .stream
                .decode_packet(data, size, &mut self.frame, &mut got_stream_frame);
        }
        let mut eof = false;
        if size == 0 {
            eof = true;
            if !self.stream.eof_done && self.frame.allocated {
                result =
                    self.stream
                        .decode_packet(data, size, &mut self.frame, &mut got_stream_frame);
            }
            eof &= self.stream.eof_done;
        }
        if self.stream.trim_start != 0 {
            self.trim_start = i32::from(self.stream.trim_start);
        }
        if self.stream.trim_end != 0 {
            self.trim_end = i32::from(self.stream.trim_end);
        }

        if got_stream_frame {
            for channel in 0..self.channels {
                self.fifo[channel].extend_from_slice(&self.frame.samples[channel]);
            }
        } else if result < 0 {
            return (result, false);
        }

        let mut got_frame = false;
        if self.stream.packet_done || self.stream.packet_loss() {
            // the one stream owns every packet
            self.stream.skip_packets = self.stream.skip_packets.saturating_sub(1);
            let mut count = self.fifo_len();
            if !eof && size != 0 {
                count -= count.min(HELD_BACK);
            }
            if (count > 0 || eof || size == 0) && !self.flushed {
                if eof {
                    let trim = (self.trim_end + self.trim_start - 128 - 64).clamp(0, count as i32);
                    count -= trim as usize;
                    self.flushed = true;
                }
                if count == 0 {
                    // ff_get_buffer of no samples
                    self.errors.error("get_buffer() failed".to_owned());
                    return (EINVAL, false);
                }
                self.output(count);
                got_frame = true;
            }
        }
        (result, got_frame)
    }
}

/// What the decoder reported, as an error.
fn invalid(decoder: &Xma) -> Error {
    let errors = decoder.stream.errors.count + decoder.errors.count;
    let first = decoder
        .stream
        .errors
        .first
        .as_deref()
        .or(decoder.errors.first.as_deref())
        // an error returned without a message: a packet FFmpeg drops as undecodable
        .unwrap_or("a packet that does not decode");
    Error::Invalid {
        errors: errors.max(1),
        first: first.to_owned(),
    }
}

/// One stream (whole packets of `channels` channels, 1 or 2) decoded with the band layout of
/// `sample_rate`: the samples interleaved.
pub(crate) fn decode_stream(
    kernel: Kernel,
    packets: &[u8],
    channels: usize,
    sample_rate: u32,
) -> Result<Vec<f32>, Error> {
    if !(1..=2).contains(&channels) {
        return Err(Error::Channels(channels));
    }
    if !packets.len().is_multiple_of(BLOCK_ALIGN) {
        return Err(Error::PartialPacket(packets.len()));
    }
    if packets.is_empty() {
        return Ok(Vec::new());
    }
    if !kernel.is_available() {
        return Err(Error::KernelUnavailable(kernel));
    }
    let rate = i32::try_from(sample_rate).map_err(|_| Error::SampleRate(sample_rate))?;
    // XMA2WAVEFORMATEX of one stream: a stream of every channel
    let stream =
        WmaPro::new(kernel, channels, rate).map_err(|first| Error::Invalid { errors: 1, first })?;
    let mut decoder = Xma {
        channels,
        stream,
        frame: StreamFrame::new(),
        fifo: [Vec::new(), Vec::new()],
        fifo_start: 0,
        trim_start: 0,
        trim_end: 0,
        flushed: false,
        skip_samples: 0,
        errors: ErrorLog::default(),
        out: Vec::with_capacity(packets.len() / BLOCK_ALIGN * 8 * SAMPLES_PER_FRAME * channels),
    };

    // the whole stream as one packet, and the padding after it
    let mut data = Vec::with_capacity(packets.len() + PADDING);
    data.extend_from_slice(packets);
    data.resize(packets.len() + PADDING, 0);
    let size = packets.len();
    let mut at = 0;
    let mut stalls = 0;
    while at < size {
        let (result, got_frame) = decoder.decode(&data[at..], size - at);
        if result < 0 {
            return Err(invalid(&decoder));
        }
        let consumed = result as usize;
        if consumed == 0 && !got_frame {
            stalls += 1;
            if stalls > STALL_LIMIT {
                return Err(Error::Invalid {
                    errors: 1,
                    first: "the decoder makes no progress".to_owned(),
                });
            }
        } else {
            stalls = 0;
        }
        at = if consumed >= size - at {
            size
        } else {
            at + consumed
        };
    }
    // draining: until a call gives no frame
    loop {
        let (result, got_frame) = decoder.decode(&data[size..], 0);
        if result < 0 {
            return Err(invalid(&decoder));
        }
        if !got_frame {
            break;
        }
    }
    if decoder.stream.errors.count + decoder.errors.count > 0 {
        return Err(invalid(&decoder));
    }
    Ok(decoder.out)
}
