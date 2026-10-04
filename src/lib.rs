// SPDX-License-Identifier: LGPL-2.1-or-later
//
// Part of xma. Ported to Rust in 2026 by the xma authors from FFmpeg's libavcodec/wmaprodec.c
// (FFmpeg n9.1-dev-56-gae4314e2f4); the original files carry these notices:
//
// Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
// Copyright (c) 2008 - 2011 Sascha Sommer, Benjamin Larsson
// Copyright (c) 2026 the xma authors (the Rust port)
//
// xma is free software; you can redistribute it and/or modify it under the terms of the GNU
// Lesser General Public License as published by the Free Software Foundation; either version
// 2.1 of the License, or (at your option) any later version.
//
// xma is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License (the LICENSE file) for more details.

//! An XMA2 decoder: FFmpeg's `xma2` decoder (`libavcodec/wmaprodec.c`, FFmpeg 9.1-dev) ported to
//! Rust, with no dependencies.
//!
//! XMA2 is the Xbox 360's audio codec, a variant of WMA Pro: a stream is a run of 2048-byte
//! packets ([`PACKET_SIZE`]), each carrying frames of 512 samples of one or two channels. A file
//! of more channels interleaves several such streams packet by packet (each packet header's skip
//! count says how many packets of the other streams follow); [`decode`] takes the packets of one
//! stream, in order.
//!
//! ```
//! # fn main() -> Result<(), xma::Error> {
//! # let packets: &[u8] = &[];
//! // the packets of one stream of two channels, encoded at 48 kHz
//! let samples: Vec<f32> = xma::decode(packets, 2, 48_000)?;
//! for frame in samples.chunks_exact(2) {
//!     let (left, right) = (frame[0], frame[1]);
//! #   let _ = (left, right);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # The samples
//!
//! The output is exactly what FFmpeg's decoder gives for the stream sent as one packet and then
//! drained, to the bit: 32-bit floats in [-1, 1], interleaved when the stream has two channels.
//! It starts after the decoder's delay (the first frame, which only primes the overlap, and 64
//! more samples), and ends with what the last frame's overlap holds, less the samples the
//! stream's frame headers trim (`trim_start + trim_end - 192`). A stream FFmpeg reports an error
//! for, or would log one for, is an [`Error::Invalid`].
//!
//! The inverse MDCT is FFmpeg's AVX2 assembly, transliterated instruction for instruction: that is
//! what FFmpeg runs on an x86-64 processor with AVX2 and FMA, and its rounding differs from
//! FFmpeg's C transform. [`Kernel::Avx2`] runs those instructions; [`Kernel::Portable`] runs
//! each one's exact arithmetic in plain Rust, on any processor. Both give the same samples.
//! [`Decoder::new`] picks the fastest.
//!
//! The tables FFmpeg computes when it starts (its sine windows, the transform's twiddles and
//! rotations, the quantizers) are held as its Windows x86-64 build computes them, so no
//! platform's maths library rounds one differently. The output therefore never depends on the
//! machine (FFmpeg's does): it is that build's output, on any processor and system.
//!
//! # Licence
//!
//! A port of FFmpeg, so under FFmpeg's licence: the GNU LGPL, version 2.1 or later.

mod bits;
mod imdct;
mod tables;
mod vlc;
mod wmapro;
mod xma;

pub use imdct::Kernel;

/// The bytes of an XMA packet.
pub const PACKET_SIZE: usize = wmapro::BLOCK_ALIGN;

/// Why a stream does not decode.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A stream has 1 or 2 channels, not this many.
    Channels(usize),
    /// This many bytes are not whole packets.
    PartialPacket(usize),
    /// A sample rate above `i32::MAX`.
    SampleRate(u32),
    /// The kernel does not run on this processor.
    KernelUnavailable(Kernel),
    /// The packets are not a valid XMA2 stream: how many errors the decoder met, and the first.
    Invalid { errors: u32, first: String },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Channels(channels) => {
                write!(f, "an XMA stream has 1 or 2 channels, not {channels}")
            }
            Error::PartialPacket(bytes) => {
                write!(
                    f,
                    "{bytes} bytes are not whole {PACKET_SIZE}-byte XMA packets"
                )
            }
            Error::SampleRate(rate) => write!(f, "a sample rate of {rate}"),
            Error::KernelUnavailable(kernel) => {
                write!(f, "this processor does not run the {kernel:?} kernel")
            }
            Error::Invalid { errors, first } => {
                write!(
                    f,
                    "not a valid XMA2 stream ({errors} errors, the first: {first})"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

/// The decoder, with a [`Kernel`]. It keeps no state between streams, so one value serves any
/// number of threads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decoder {
    kernel: Kernel,
}

impl Default for Decoder {
    fn default() -> Decoder {
        Decoder::new()
    }
}

impl Decoder {
    /// The decoder with the fastest kernel this processor runs ([`Kernel::detect`]).
    pub fn new() -> Decoder {
        Decoder {
            kernel: Kernel::detect(),
        }
    }

    /// The decoder with `kernel`, if this processor runs it.
    pub fn with_kernel(kernel: Kernel) -> Result<Decoder, Error> {
        if kernel.is_available() {
            Ok(Decoder { kernel })
        } else {
            Err(Error::KernelUnavailable(kernel))
        }
    }

    /// Its kernel.
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// One stream: `packets` are its whole packets in order, `channels` its channels (1 or 2),
    /// `sample_rate` the rate it was encoded at, which selects its band layout: that of the first
    /// of 24000, 32000, 44100 and 48000 Hz at or above it (48000 above that). The samples
    /// interleaved; no packets, no samples.
    pub fn decode(
        &self,
        packets: &[u8],
        channels: usize,
        sample_rate: u32,
    ) -> Result<Vec<f32>, Error> {
        xma::decode_stream(self.kernel, packets, channels, sample_rate)
    }
}

/// One stream with [`Decoder::new`]'s decoder (see [`Decoder::decode`]).
pub fn decode(packets: &[u8], channels: usize, sample_rate: u32) -> Result<Vec<f32>, Error> {
    Decoder::new().decode(packets, channels, sample_rate)
}

#[cfg(test)]
mod tests;
