#include "citadel/move.hpp"

#include <algorithm>
#include <cctype>

namespace citadel {

std::string coordToString(std::uint8_t s) {
  if (s == SQ_NONE) return "--";
  const int r = row(s);
  const int c = col(s);
  const char file = static_cast<char>('A' + c);
  const char rank = static_cast<char>('0' + (N - r)); // r=0 -> '9'
  return std::string{file, rank};
}

std::optional<std::uint8_t> parseCoord(std::string_view sv) {
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) sv.remove_prefix(1);
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) sv.remove_suffix(1);
  if (sv.size() != 2) return std::nullopt;

  const char f0 = static_cast<char>(std::toupper(static_cast<unsigned char>(sv[0])));
  const char r0 = sv[1];
  if (f0 < 'A' || f0 >= static_cast<char>('A' + N)) return std::nullopt;
  if (r0 < '1' || r0 > '9') return std::nullopt;

  const int c = f0 - 'A';
  const int rank = r0 - '0';
  const int r = N - rank;
  if (!inBounds(r, c)) return std::nullopt;
  return sq(r, c);
}

std::string moveToString(const Move& m) {
  switch (m.type) {
    case MoveType::Normal:
      return coordToString(m.from) + coordToString(m.to);
    case MoveType::MasonConstruct:
      return "con " + coordToString(m.from) + "@" + coordToString(m.to);
    case MoveType::MasonCommand: {
      std::string s = "cmd " + coordToString(m.from) + coordToString(m.to);
      if (m.aux1 != SQ_NONE) s += "@" + coordToString(m.aux1);
      return s;
    }
    case MoveType::CatapultMove: {
      std::string s = "cat " + coordToString(m.from) + coordToString(m.to);
      if (m.aux1 != SQ_NONE) s += "x" + coordToString(m.aux1);
      return s;
    }
    case MoveType::CatapultRangedDemolish:
      return "rd " + coordToString(m.from) + "x" + coordToString(m.to);
    case MoveType::Bastion:
      return "bas " + coordToString(m.from) + "<>" + coordToString(m.to) + "@" + coordToString(m.aux1) + "," +
             coordToString(m.aux2);
    default:
      return "??";
  }
}

std::string moveToPgnToken(const Move& m) {
  std::string s = moveToString(m);
  s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
  return s;
}

} // namespace citadel

