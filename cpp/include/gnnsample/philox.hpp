// Philox-4x32-10 counter-based RNG. Must match python/gnnsample/philox.py word for word:
// the reference sampler and the GPU sampler draw the same numbers for the same
// (seed, hop, node, draw), independent of how work is mapped to threads.
#pragma once

#include <cstdint>

namespace gnnsample {

struct Philox4x32 {
  uint32_t c[4];
};

inline Philox4x32 philox4x32(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0,
                             uint32_t k1) {
  constexpr uint64_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
  constexpr uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
#pragma unroll
  for (int r = 0; r < 10; ++r) {
    if (r) {
      k0 += W0;
      k1 += W1;
    }
    const uint64_t p0 = M0 * c0, p1 = M1 * c2;
    const uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ c1 ^ k0;
    const uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ c3 ^ k1;
    c1 = static_cast<uint32_t>(p1);
    c3 = static_cast<uint32_t>(p0);
    c0 = n0;
    c2 = n2;
  }
  return {{c0, c1, c2, c3}};
}

// Random word `draw` for `node` at `hop`: counter (node, hop, draw / 4, 0), key = seed.
inline uint32_t draw_u32(uint64_t seed, uint32_t hop, uint32_t node, uint32_t draw) {
  const Philox4x32 w = philox4x32(node, hop, draw >> 2, 0, static_cast<uint32_t>(seed),
                                  static_cast<uint32_t>(seed >> 32));
  return w.c[draw & 3];
}

// Multiply-shift range reduction to [0, n).
inline int64_t bounded(uint32_t r, uint64_t n) {
  return static_cast<int64_t>((static_cast<uint64_t>(r) * n) >> 32);
}

}  // namespace gnnsample
