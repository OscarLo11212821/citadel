#pragma once

#include <cstdint>
#include <string>

#include "citadel/core.hpp"

namespace citadel {

struct Move {
  MoveType type;
  std::uint8_t from;
  std::uint8_t to;
  std::uint8_t aux1; // build/demolish/wall1
  std::uint8_t aux2; // wall2 (Bastion)
};

[[nodiscard]] constexpr Move nullMove() { return Move{MoveType::Normal, SQ_NONE, SQ_NONE, SQ_NONE, SQ_NONE}; }

[[nodiscard]] std::string moveToString(const Move& m);
// PGN tokens must be whitespace-free; this is a Citadel-specific "SAN-like" token.
[[nodiscard]] std::string moveToPgnToken(const Move& m);

} // namespace citadel

