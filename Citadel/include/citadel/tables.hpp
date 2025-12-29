#pragma once

#include <array>
#include <cstdint>

#include "citadel/core.hpp"

namespace citadel {

// Direction indices used by Tables::ray / Tables::rayLen
// 0:N, 1:S, 2:W, 3:E, 4:NW, 5:NE, 6:SW, 7:SE
constexpr std::array<Coord, 8> DIRS8 = {{
  {-1, 0},
  {1, 0},
  {0, -1},
  {0, 1},
  {-1, -1},
  {-1, 1},
  {1, -1},
  {1, 1},
}};

constexpr std::array<Coord, 4> DIRS4 = {{
  {-1, 0},
  {1, 0},
  {0, -1},
  {0, 1},
}};

constexpr std::array<Coord, 8> KNIGHT = {{
  {-2, -1},
  {-2, 1},
  {-1, -2},
  {-1, 2},
  {1, -2},
  {1, 2},
  {2, -1},
  {2, 1},
}};

struct Tables {
  std::array<std::uint8_t, SQ_N> isKeep{};

  std::array<std::uint8_t, SQ_N> knightCount{};
  std::array<std::array<std::uint8_t, 8>, SQ_N> knightTargets{}; // max 8

  std::array<std::uint8_t, SQ_N> kingCount{};
  std::array<std::array<std::uint8_t, 8>, SQ_N> kingTargets{}; // max 8

  // Rays for sliding pieces. Each direction holds up to 8 squares.
  std::array<std::array<std::uint8_t, 8>, SQ_N> rayLen{};
  std::array<std::array<std::array<std::uint8_t, 8>, 8>, SQ_N> ray{};

  // Zobrist keys
  uint64_t pieceKeys[2][6][SQ_N]{};
  uint64_t wallKeys[2][2][SQ_N]{};
  uint64_t turnKey{};
  uint64_t bastionKeys[2]{};
  uint64_t wallBuiltLastKeys[2]{};
};

[[nodiscard]] const Tables& tables();

} // namespace citadel

