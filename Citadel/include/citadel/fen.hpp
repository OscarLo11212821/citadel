#pragma once

#include <string>
#include <string_view>

#include "citadel/position.hpp"

namespace citadel {

// Convenience free-functions (wrapping Position methods).
[[nodiscard]] inline std::string toFEN(const Position& p) { return p.toFEN(); }
[[nodiscard]] inline Position fromFEN(std::string_view fen) { return Position::fromFEN(fen); }

} // namespace citadel

