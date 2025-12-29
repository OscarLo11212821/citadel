#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace citadel {

constexpr int N = 9;
constexpr int SQ_N = N * N; // 81
constexpr std::uint8_t SQ_NONE = 0xFF;

enum class Color : std::uint8_t { White = 0, Black = 1 };

[[nodiscard]] constexpr Color other(Color c) {
  return (c == Color::White) ? Color::Black : Color::White;
}

enum class PieceType : std::uint8_t { Mason = 0, Catapult, Lancer, Pegasus, Minister, Sovereign, Count };

enum class MoveType : std::uint8_t {
  Normal,
  MasonConstruct,
  MasonCommand,          // move + optional build
  CatapultMove,          // move + optional adjacent demolish
  CatapultRangedDemolish,
  Bastion,               // swap + place 2 walls
};

enum class WinReason : std::uint8_t { None = 0, Regicide, Entombment };

struct Coord {
  int r = 0;
  int c = 0;
};

[[nodiscard]] constexpr bool inBounds(int r, int c) {
  return r >= 0 && r < N && c >= 0 && c < N;
}

[[nodiscard]] constexpr std::uint8_t sq(int r, int c) {
  return static_cast<std::uint8_t>(r * N + c);
}

[[nodiscard]] constexpr int row(std::uint8_t s) {
  return static_cast<int>(s) / N;
}

[[nodiscard]] constexpr int col(std::uint8_t s) {
  return static_cast<int>(s) % N;
}

[[nodiscard]] constexpr bool isKeep(int r, int c) {
  return r >= 3 && r <= 5 && c >= 3 && c <= 5;
}

[[nodiscard]] constexpr bool isKeepSq(std::uint8_t s) {
  return (s != SQ_NONE) && isKeep(row(s), col(s));
}

[[nodiscard]] std::string coordToString(std::uint8_t s);
[[nodiscard]] std::optional<std::uint8_t> parseCoord(std::string_view s);

[[nodiscard]] constexpr std::string_view colorName(Color c) {
  return (c == Color::White) ? "White" : "Black";
}

[[nodiscard]] constexpr std::string_view pieceName(PieceType p) {
  switch (p) {
    case PieceType::Mason: return "Mason";
    case PieceType::Catapult: return "Catapult";
    case PieceType::Lancer: return "Lancer";
    case PieceType::Pegasus: return "Pegasus";
    case PieceType::Minister: return "Minister";
    case PieceType::Sovereign: return "Sovereign";
    default: return "?";
  }
}

} // namespace citadel

