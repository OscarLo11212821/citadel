#pragma once

#include <cstdint>
#include <vector>

#include "citadel/move.hpp"
#include "citadel/position.hpp"

namespace citadel {

struct PerftStats {
  std::uint64_t nodes = 0;
  double seconds = 0.0;
  double nps = 0.0;
};

[[nodiscard]] std::uint64_t perft(Position& pos, int depth);
[[nodiscard]] std::vector<std::pair<Move, std::uint64_t>> perftDivide(Position& pos, int depth);
[[nodiscard]] PerftStats perftTimed(Position& pos, int depth);

} // namespace citadel

