# xma-rs [![Build Status]][actions]

[Build Status]: https://github.com/HaydnTrigg/xma-rs/actions/workflows/build.yaml/badge.svg
[actions]: https://github.com/HaydnTrigg/xma-rs/actions

> [!WARNING]
> This is an AI-generated conversion of FFmpeg's C code to Rust.
> This project is a work in progress and the API subject to change.

A Rust decoder for XMA2, the audio format of the Xbox 360. It is a line-by-line port of
[FFmpeg](https://ffmpeg.org)'s XMA2 decoder and produces exactly the same samples as FFmpeg, bit
for bit. It has no dependencies, and the only `unsafe` code is in the inverse MDCT.

XMA2 is a variant of Microsoft's WMA Pro codec.

## Minimum Rust version

1.88.

## Usage

```toml
[dependencies]
xma = { git = "https://github.com/HaydnTrigg/xma-rs" }
```

### API

#### `xma::decode`

```rust
pub fn decode(packets: &[u8], channels: usize, sample_rate: u32) -> Result<Vec<f32>, xma::Error>
```

Decodes one stream with the fastest kernel this processor runs.

- `packets`: the stream's whole packets, in order. Each packet is `xma::PACKET_SIZE` (2048)
  bytes. A file of more than two channels interleaves several streams packet by packet, soseparate them first.
- `channels`: 1 or 2.
- `sample_rate`: the rate the stream was encoded at. It selects the band layout: that of the first of 24000, 32000, 44100 and 48000 Hz.

Returns 32-bit float samples in [-1, 1], interleaved when there are two channels. No packets
give no samples.

### Example

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The raw packets of one stream (no RIFF header): two channels, encoded at 48 kHz.
    let packets = std::fs::read("stream.bin")?;

    let samples = xma::decode(&packets, 2, 48_000)?;

    // Two channels come interleaved: left, right, left, right, ...
    for frame in samples.chunks_exact(2) {
        let (left, right) = (frame[0], frame[1]);
        println!("{left} {right}");
    }

    Ok(())
}
```

## Speed

Against FFmpeg's decoder through its C API, on a Core Ultra 7 165H:

| | FFmpeg | xma, AVX2 | xma, portable |
|---|---|---|---|
| 172 long streams (256 packets or more), one thread | 12.8 s | 10.0 s | 14.2 s |
| 1 in 8 of all streams, one thread | 21.1 s | 8.2 s | 13.3 s |
| all 43,729 streams, 22 threads | 37.4 s | 15.7 s | 28.2 s |

## License and credits

xma is a port of FFmpeg's LGPL code and is licensed under the same terms,
LGPL-2.1-or-later ([LICENSE](LICENSE)). The decoder is the work of the
[FFmpeg developers](https://github.com/FFmpeg/FFmpeg/blob/master/CREDITS); each source file names
the FFmpeg file it was ported from and keeps its copyright notices. Refer to FFmpeg's
[LICENSE.md](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md) for detailed information.
