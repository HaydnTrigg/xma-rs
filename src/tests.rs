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

use super::*;

#[test]
fn an_empty_stream_decodes_to_nothing() {
    assert_eq!(decode(&[], 1, 44_100), Ok(Vec::new()));
}

#[test]
fn what_is_not_xma_is_an_error() {
    assert_eq!(decode(&[0; 100], 1, 44_100), Err(Error::PartialPacket(100)));
    assert_eq!(
        decode(&[0; PACKET_SIZE], 3, 44_100),
        Err(Error::Channels(3))
    );
    // packets of noise: the decoder reports them, it does not return silence
    let mut noise = vec![0u8; PACKET_SIZE * 4];
    let mut state = 0x1234_5678u32;
    for byte in &mut noise {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *byte = (state >> 24) as u8;
    }
    for packet in noise.as_chunks_mut::<PACKET_SIZE>().0 {
        packet[0] = 3 << 2; // three frames
        packet[3] = 0;
    }
    for kernel in [Kernel::Avx2, Kernel::Portable] {
        if let Ok(decoder) = Decoder::with_kernel(kernel) {
            let result = decoder.decode(&noise, 2, 48_000);
            assert!(
                matches!(result, Err(Error::Invalid { .. })),
                "{:?}",
                result.map(|samples| samples.len())
            );
        }
    }
}

#[test]
fn a_kernel_the_processor_lacks_is_refused() {
    assert!(Decoder::with_kernel(Kernel::Portable).is_ok());
    assert_eq!(
        Decoder::with_kernel(Kernel::Avx2).is_ok(),
        Kernel::Avx2.is_available()
    );
    assert!(Decoder::new().kernel().is_available());
}
