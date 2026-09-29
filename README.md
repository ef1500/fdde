# FDDE — Fixed Depth Differential Encoder

Turn a continuous signal into a stream of small integers describing its **local shape**, not its values.

FDDE keeps only whether each step went up or not-up, slides a fixed-width window over those direction bits, and packs the window positions into tokens (naive hashes) you can hash with a proper algorithm, count, index or diff. The output is exactly invariant to amplitude, costs one compare and one shift per step, and lands in a small discrete alphabet — which makes waveform data behave (sorta) like text.

---

## The idea in one table

`dx=1, depth=4`, over a sampled sine:

| Slice | Mean | Direction | Residual |
|-------|------|-----------|----------|
| 0  | 0.0     | — hidden | — hidden |
| 1  | 0.5538  | up   | `0001` |
| 2  | -0.3698 | down | `0010` |
| 3  | -0.9534 | down | `0100` |
| 4  | -0.6604 | up   | `1001` |
| 5  | 0.2397  | up   | `0011` |
| 6  | 0.9195  | up   | `0111` |
| 7  | 0.7539  | down | `1110` |
| 8  | -0.1048 | down | `1100` |
| 9  | -0.8672 | down | `1000` |
| 10 | -0.9589 | down | `0000` |

Read **down the last column** for the direction sequence; read **across a row** for the last four directions as one number. The first slice has no predecessor, so it produces no residual. It is labeled as hidden to convey that.

At `size=4`, four residuals concatenate per token:

```
0001 0010 0100 1001  ->  0x1249
0011 0111 1110 1100  ->  0x37ec
1000 0000            ->  0x8000   (short group, zero-padded at the back)
```

---

## Quick start

Standard library only, Python 3.7+.

```bash
git clone https://github.com/ef1500/fdde
cd fdde/src
python fdde.py
```

```python
from fdde import FDDE

fdde = FDDE(dx=1, depth=4, size=4)
fdde.encode([0, 0.5538, -0.3698, -0.9534, -0.6604,
             0.2397, 0.9195, 0.7539, -0.1048, -0.8672, -0.9589])
# [4681, 14316, 32768]   i.e. 0x1249, 0x37ec, 0x8000
```

---

## How it works

1. **Slice** the input into chunks of `dx` samples (no-op at `dx=1`).
2. **Average** each chunk to its mean. This is the decimation stage.
3. **Compare** adjacent means: `1` if `curr > prev`, else `0`. `N` means give `N-1` bits.
4. **Window and pack.** A `depth`-bit shift register holds the last `depth` direction bits, newest in the low bit, emitting its value after each new bit. Consecutive step residuals overlap by `depth-1` bits — that overlap is what makes this a sliding window rather than a partition. Every `size` step residuals concatenate into one token of `size * depth` bits; a short final group is zero-padded at the back.

For `N` samples: `means = ceil(N/dx)`, `steps = means - 1`, `residuals = ceil(steps/size)`.

---

## Parameters

| Param | Meaning | Raising it |
|-------|---------|------------|
| `dx` | Samples averaged per slice | Coarser time resolution, more noise immunity, fewer residuals |
| `depth` | Bits of history per window | More context per token; alphabet grows as `2^depth` |
| `size` | Step residuals per token | Wider, rarer tokens; fewer of them |

### Sizing the alphabet

A token has `size * depth` bits, but because step residuals overlap by `depth-1` bits, a token is determined by `depth-1` bits of history plus `size` new bits:

```
reachable tokens = 2^(depth + size - 1)
```

At `depth=4, size=4` that is **128 of 65536 representable values** — a 512× overcount if you size structures from the bit width. For anything frequency-based, keep `depth + size - 1` comfortably above `log2(number of tokens)`. `depth=8, size=13` is a solid working default.

---

## Properties

- **Amplitude invariance is exact.** Gain, offset and affine transforms leave residuals bit-identical. At `dx=1` any strictly increasing transform is invisible; at `dx>1` the guarantee narrows to positive affine maps, since a mean commutes with `a*x+b` but not with a nonlinear map.
- **Turning points are preserved exactly.** A strict extremum of the input at index `k` is a strict extremum of the reconstruction at `k-1`, always — structural, not statistical, so it holds even for white noise.
- **Every column is the same bitstream, shifted.** Each direction bit is written `depth` times, so tokens can be lost and still recovered from neighbours' overlapping windows. At `depth=8, size=2`, dropping 20% of tokens leaves 99.96% of direction bits recoverable. Keep `size < depth` for near-lossless behaviour over a lossy channel.
- **Reading columns reconstructs the curve.** Integrate the direction bits (`+1`/`-1`). Detrend the result first — ties encode as down, and the resulting bit imbalance integrates into a ramp that swamps the shape. On smooth, oversampled, low-noise signals this recovers shape at `r ≈ 0.99`.

---

## Fingerprinting with hapax residuals

Encode a waveform, count residual frequencies, and keep the residuals appearing **exactly once** — the *hapax legomena*. Common shapes recur and say nothing; a residual firing once is a distinctive moment in that recording.

```python
import collections
from fdde import FDDE

def hapax(signal, dx=16, depth=8, size=13):
    res = FDDE(dx=dx, depth=depth, size=size).encode(list(signal))
    res = res[:-1]                     # drop the zero-padded tail token
    counts = collections.Counter(res)
    return {r for r in res if counts[r] == 1}

similarity = len(a & b) / len(a | b)   # Jaccard
```

On 30 s excerpts of real music at 22050 Hz, `dx=64, depth=8, size=13`:

| Comparison | Jaccard |
|---|---:|
| gain ×0.4 | **1.0000** |
| 64 kbps mp3 round-trip | 0.6067 |
| noise @ 20 dB SNR | 0.7322 |
| unrelated tracks | 0.0009 – 0.0135 |

A gain change is bit-identical; lossy compression and added noise stay well clear of the floor. Larger `dx` buys robustness.

### Alignment

Token boundaries are anchored to the input start, so an unaligned query lands in a different grouping and matches nothing. Index the database at offset 0 and encode each **query** at several offsets, taking the best score.

In a 35-segment retrieval benchmark with mp3-degraded, randomly-shifted queries, sweeping slice offsets alone gave 7/12 top-1; sweeping **both slice and sub-slice offsets** (52 encodings per query) gave **12/12**, with the true match scoring 11× the best wrong one. Raising `dx` is not a substitute — it lifts the wrong-match floor too.

---

## Uses

Built for, and measured on, audio fingerprinting and vocal codebooks (tally residual frequencies, keep the top-`k` as a speaker profile — untouched by mic gain, distance and recording level). The same properties suit:

- Query-by-humming and melodic contour search
- Motif and repeat discovery in time series
- Gesture and IMU recognition, where scale varies per person and mounting
- Biosignal beat tokenization (ECG, PPG)
- Change-point and anomaly detection — rare tokens as novelty flags
- Near-duplicate detection at scale, via minhash/LSH over the token stream
- Feeding continuous signals to sequence models without a learned quantizer
- Embedded classification: integer-only inner loop, no transform
- Lossy-channel telemetry, using the erasure tolerance above

### Not useful for

- **Compression.** It spends `depth` bits per step on one bit of information. The redundancy is the product, not waste.
- **Recovering amplitude.** Reconstruction returns shape and turning points, never magnitude, and degrades quickly with noise.
- **Polarity-inverted signals.** Multiplying by −1 flips every bit; nothing survives.
- **Multi-dimensional data.** `encode` takes a 1-D sequence; encode channels separately.

---

## Caveats

- **The zero-padded tail token is always a spurious hapax.** It is unreachable by normal encoding, so it occurs exactly once by construction. Drop the last token before any frequency analysis.
- **Ties encode as down.** Silence and constant input collapse to a single token and a monotonically falling reconstruction. Strip silence, dither, or raise `dx`. Coarsely quantized input is nearly as bad.
- **The first `depth-1` residuals are contaminated** by the shift register's assumed-zero start. Discard them when fingerprinting.
- **Near-flat regions are noise-decided**, since the sign of a tiny difference is arbitrary.

---

## API

```python
FDDE(dx: int, depth: int, size: int)   # all >= 1, else ValueError
```

| Method | Returns |
|---|---|
| `encode(data)` | `List[int]` — the FDDE residuals. Main entry point. `[]` for single-sample input. |
| `slice_data(data)` | Input cut into `dx`-sized chunks (raw list when `dx == 1`). Raises on empty input. |
| `mean(chunk)` | Mean of one chunk. Raises if empty or longer than `dx`. |
| `list_means(slices)` | Chunks reduced to means. A flat float list passes through, but only if `dx == 1`. |

---

## Related work

- **Parsons code** (Parsons, 1975) — up/down/repeat melodic contour, the classic query-by-humming basis. The direction-bit stage without the sliding window.
- **Delta and sign-delta modulation** — the same 1-bit-per-step quantization from a signal-processing angle.
- **Haitsma–Kalker fingerprinting** (2002) — sub-fingerprints from the sign of energy differences, matched on Hamming distance. The same invariance insight applied to a spectrogram.
- **Shingling and minhash** (Broder, 1997) — turns an overlapping-window token stream into a similarity search; composes directly with hapax sets.

I pretty much just added differentials to Parson's idea, and then used a sliding window to represent it, which can be used to create a set of naive hashes. I use this in my personal, local similar music search.

---

## Credit

Christopher J. Cole — <https://github.com/ef1500/fdde>

```bibtex
@software{cole_fdde,
  author = {Cole, Christopher J.},
  title  = {FDDE: Fixed Depth Differential Encoder},
  year   = {2026},
  url    = {https://github.com/ef1500/fdde}
}
```

---
## License

[MIT](LICENSE.txt) © 2026 Christopher J. Cole.
