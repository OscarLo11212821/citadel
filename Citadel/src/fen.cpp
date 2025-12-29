#include "citadel/position.hpp"

#include <cctype>
#include <sstream>
#include <stdexcept>

namespace citadel {

std::string Position::toFEN() const {
  std::ostringstream oss;

  for (int r = 0; r < N; ++r) {
    int empty = 0;
    for (int c = 0; c < N; ++c) {
      const std::uint8_t s = sq(r, c);
      const std::int8_t v = at(s);
      if (v == 0) {
        ++empty;
        continue;
      }
      if (empty) {
        oss << empty;
        empty = 0;
      }

      char ch = '?';
      if (isPieceVal(v)) {
        switch (pieceOf(v)) {
          case PieceType::Mason: ch = 'M'; break;
          case PieceType::Catapult: ch = 'C'; break;
          case PieceType::Lancer: ch = 'L'; break;
          case PieceType::Pegasus: ch = 'P'; break;
          case PieceType::Minister: ch = 'I'; break;
          case PieceType::Sovereign: ch = 'S'; break;
          default: ch = '?'; break;
        }
      } else if (isWallVal(v)) {
        ch = (wallHp(v) == 2) ? 'R' : 'W';
      }
      if (v < 0) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      oss << ch;
    }
    if (empty) oss << empty;
    if (r != N - 1) oss << '/';
  }

  oss << ' ' << ((turn_ == Color::White) ? 'w' : 'b') << ' ';

  std::string rights;
  if (bastionRight_[static_cast<int>(Color::White)]) rights.push_back('B');
  if (bastionRight_[static_cast<int>(Color::Black)]) rights.push_back('b');
  if (rights.empty()) rights = "-";
  oss << rights << ' ';

  std::string wallSeq;
  if (wallBuiltLast(Color::White)) wallSeq.push_back('w');
  if (wallBuiltLast(Color::Black)) wallSeq.push_back('b');
  if (wallSeq.empty()) wallSeq = "-";
  oss << wallSeq << ' ' << halfmove_ << ' ' << fullmove_;

  return oss.str();
}

Position Position::fromFEN(std::string_view fen) {
  std::istringstream iss{std::string(fen)};
  std::string boardStr, turnStr;
  if (!(iss >> boardStr >> turnStr)) throw std::runtime_error("Invalid FEN: expected <board> <turn> ...");

  std::string rightsStr = "Bb";
  std::string wallStr = "-";
  int halfmove = 0;
  int fullmove = 1;

  auto isDigits = [](std::string_view s) -> bool {
    if (s.empty()) return false;
    for (const char ch : s) {
      if (ch < '0' || ch > '9') return false;
    }
    return true;
  };

  std::string tok;
  if (iss >> tok) {
    rightsStr = tok;
    if (iss >> tok) {
      if (isDigits(tok)) {
        halfmove = std::atoi(tok.c_str());
        wallStr = "-";
      } else {
        wallStr = tok;
        if (!(iss >> halfmove)) halfmove = 0;
      }
      if (!(iss >> fullmove)) fullmove = 1;
    }
  }

  Position p;
  p.b_.fill(0);
  p.winner_ = SQ_NONE;
  p.winReason_ = WinReason::None;

  if (turnStr.empty()) throw std::runtime_error("Invalid FEN: empty turn field");
  const char t = static_cast<char>(std::tolower(static_cast<unsigned char>(turnStr[0])));
  if (t == 'w') p.turn_ = Color::White;
  else if (t == 'b') p.turn_ = Color::Black;
  else throw std::runtime_error("Invalid FEN: turn must be 'w' or 'b'");

  p.bastionRight_[static_cast<int>(Color::White)] = false;
  p.bastionRight_[static_cast<int>(Color::Black)] = false;
  if (rightsStr != "-") {
    for (const char rc : rightsStr) {
      if (rc == 'B') p.bastionRight_[static_cast<int>(Color::White)] = true;
      if (rc == 'b') p.bastionRight_[static_cast<int>(Color::Black)] = true;
    }
  }

  p.wallBuiltLast_[static_cast<int>(Color::White)] = false;
  p.wallBuiltLast_[static_cast<int>(Color::Black)] = false;
  if (wallStr != "-") {
    for (const char rc : wallStr) {
      const char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(rc)));
      if (lc == 'w') p.wallBuiltLast_[static_cast<int>(Color::White)] = true;
      if (lc == 'b') p.wallBuiltLast_[static_cast<int>(Color::Black)] = true;
    }
  }

  p.halfmove_ = halfmove;
  p.fullmove_ = fullmove;

  int r = 0;
  int c = 0;
  for (const char raw : boardStr) {
    if (raw == '/') {
      if (c != N) throw std::runtime_error("Invalid FEN: rank does not have 9 files");
      ++r;
      c = 0;
      continue;
    }
    if (r >= N) throw std::runtime_error("Invalid FEN: too many ranks");

    if (raw >= '1' && raw <= '9') {
      c += raw - '0';
      if (c > N) throw std::runtime_error("Invalid FEN: file overflow");
      continue;
    }

    const bool isWhite = std::isupper(static_cast<unsigned char>(raw)) != 0;
    const char ch = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
    if (c >= N) throw std::runtime_error("Invalid FEN: too many files in rank");

    const Color col = isWhite ? Color::White : Color::Black;
    std::int8_t v = 0;
    switch (ch) {
      case 'M': v = Position::makePiece(col, PieceType::Mason); break;
      case 'C': v = Position::makePiece(col, PieceType::Catapult); break;
      case 'L': v = Position::makePiece(col, PieceType::Lancer); break;
      case 'P': v = Position::makePiece(col, PieceType::Pegasus); break;
      case 'I': v = Position::makePiece(col, PieceType::Minister); break;
      case 'S': v = Position::makePiece(col, PieceType::Sovereign); break;
      case 'W': v = Position::makeWall(col, 1); break;
      case 'R': v = Position::makeWall(col, 2); break;
      default: throw std::runtime_error(std::string("Invalid FEN: unknown piece '") + raw + "'");
    }

    p.b_[sq(r, c)] = v;
    ++c;
  }

  if (r != N - 1 || c != N) throw std::runtime_error("Invalid FEN: board must be 9 ranks of 9 files");

  p.rebuildDerived();
  return p;
}

} // namespace citadel

