#include "citadel/tables.hpp"

namespace citadel {

static std::uint64_t splitmix64(std::uint64_t& x) {
  // Deterministic PRNG (good enough for Zobrist keys).
  // Public domain reference: splitmix64 by Sebastiano Vigna.
  x += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static Tables buildTables() {
  Tables t{};

  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < N; ++c) {
      const std::uint8_t s = sq(r, c);
      t.isKeep[s] = static_cast<std::uint8_t>(isKeep(r, c) ? 1 : 0);

      // Knight targets
      {
        std::uint8_t n = 0;
        for (const auto& d : KNIGHT) {
          const int rr = r + d.r;
          const int cc = c + d.c;
          if (!inBounds(rr, cc)) continue;
          t.knightTargets[s][n++] = sq(rr, cc);
        }
        t.knightCount[s] = n;
      }

      // King targets (8-adjacent)
      {
        std::uint8_t n = 0;
        for (const auto& d : DIRS8) {
          const int rr = r + d.r;
          const int cc = c + d.c;
          if (!inBounds(rr, cc)) continue;
          t.kingTargets[s][n++] = sq(rr, cc);
        }
        t.kingCount[s] = n;
      }

      // Rays (8 directions)
      for (std::uint8_t dir = 0; dir < 8; ++dir) {
        const auto& d = DIRS8[dir];
        std::uint8_t len = 0;
        int rr = r + d.r;
        int cc = c + d.c;
        while (inBounds(rr, cc)) {
          t.ray[s][dir][len++] = sq(rr, cc);
          rr += d.r;
          cc += d.c;
        }
        t.rayLen[s][dir] = len;
      }
    }
  }

  // Zobrist keys (used by Position::hash_/repetition detection).
  // Note: Search has its own independent Zobrist in search.cpp for the TT.
  std::uint64_t seed = 0xC17ADE10A5F00D42ull; // arbitrary fixed seed (deterministic builds)
  for (int color = 0; color < 2; ++color) {
    for (int pt = 0; pt < 6; ++pt) {
      for (std::uint8_t s = 0; s < SQ_N; ++s) {
        t.pieceKeys[color][pt][s] = splitmix64(seed);
      }
    }
    for (int hpIdx = 0; hpIdx < 2; ++hpIdx) {
      for (std::uint8_t s = 0; s < SQ_N; ++s) {
        t.wallKeys[color][hpIdx][s] = splitmix64(seed);
      }
    }
  }
  t.turnKey = splitmix64(seed);
  t.bastionKeys[0] = splitmix64(seed);
  t.bastionKeys[1] = splitmix64(seed);
  t.wallBuiltLastKeys[0] = splitmix64(seed);
  t.wallBuiltLastKeys[1] = splitmix64(seed);

  return t;
}

const Tables& tables() {
  static const Tables t = buildTables();
  return t;
}

} // namespace citadel

