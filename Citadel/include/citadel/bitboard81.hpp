#pragma once

#include <bit>
#include <cstdint>

#include "citadel/core.hpp"

namespace citadel {

struct Bitboard81 {
  std::uint64_t lo = 0; // squares 0..63
  std::uint64_t hi = 0; // squares 64..80 stored in bits 0..16

  [[nodiscard]] constexpr bool empty() const { return (lo | hi) == 0; }
  [[nodiscard]] constexpr bool any() const { return !empty(); }

  [[nodiscard]] constexpr bool test(std::uint8_t s) const {
    if (s == SQ_NONE) return false;
    if (s < 64) return (lo >> s) & 1ULL;
    return (hi >> (s - 64)) & 1ULL;
  }

  constexpr void set(std::uint8_t s) {
    if (s == SQ_NONE) return;
    if (s < 64) lo |= (1ULL << s);
    else hi |= (1ULL << (s - 64));
  }

  constexpr void reset(std::uint8_t s) {
    if (s == SQ_NONE) return;
    if (s < 64) lo &= ~(1ULL << s);
    else hi &= ~(1ULL << (s - 64));
  }

  [[nodiscard]] constexpr std::uint32_t popcount() const {
    return static_cast<std::uint32_t>(std::popcount(lo) + std::popcount(hi));
  }

  [[nodiscard]] constexpr Bitboard81 operator|(Bitboard81 o) const { return Bitboard81{lo | o.lo, hi | o.hi}; }
  [[nodiscard]] constexpr Bitboard81 operator&(Bitboard81 o) const { return Bitboard81{lo & o.lo, hi & o.hi}; }
  [[nodiscard]] constexpr Bitboard81 operator^(Bitboard81 o) const { return Bitboard81{lo ^ o.lo, hi ^ o.hi}; }

  constexpr Bitboard81& operator|=(Bitboard81 o) {
    lo |= o.lo;
    hi |= o.hi;
    return *this;
  }
  constexpr Bitboard81& operator&=(Bitboard81 o) {
    lo &= o.lo;
    hi &= o.hi;
    return *this;
  }
  constexpr Bitboard81& operator^=(Bitboard81 o) {
    lo ^= o.lo;
    hi ^= o.hi;
    return *this;
  }

  [[nodiscard]] constexpr bool operator==(Bitboard81 o) const { return lo == o.lo && hi == o.hi; }
  [[nodiscard]] constexpr bool operator!=(Bitboard81 o) const { return !(*this == o); }

  // Pops and returns the least-significant set square index. Requires any()==true.
  [[nodiscard]] inline std::uint8_t pop_lsb() {
    if (lo) {
      const int bit = std::countr_zero(lo);
      lo &= (lo - 1);
      return static_cast<std::uint8_t>(bit);
    }
    const int bit = std::countr_zero(hi);
    hi &= (hi - 1);
    return static_cast<std::uint8_t>(bit + 64);
  }
};

} // namespace citadel

