#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "citadel/bitboard81.hpp"
#include "citadel/core.hpp"
#include "citadel/move.hpp"

namespace citadel {

struct MoveList {
  std::array<Move, 4096> buf;
  std::uint32_t size = 0;

  void clear() { size = 0; }
  [[nodiscard]] bool empty() const { return size == 0; }
  void push(const Move& m) {
    assert(size < buf.size());
    buf[size++] = m;
  }
};

struct Undo {
  // Restored wholesale (not derived from setSquareRaw).
  Color prevTurn = Color::White;
  bool prevBastionRight[2]{true, true};
  bool prevWallBuiltLast[2]{false, false};
  std::uint8_t prevSovereignSq[2]{SQ_NONE, SQ_NONE};
  int prevWallTokens[2]{0, 0};
  int prevHalfmove = 0;
  int prevFullmove = 1;
  std::uint8_t prevWinner = SQ_NONE; // SQ_NONE means none, otherwise Color as 0/1
  WinReason prevWinReason = WinReason::None;

  // Changed squares and their previous values.
  std::array<std::uint8_t, 6> sq{};
  std::array<std::int8_t, 6> prev{};
  std::uint8_t sqCount = 0;
};

// Search-only helper: a "pass" move used for null-move pruning.
struct NullUndo {
  Color prevTurn = Color::White;
  int prevFullmove = 1;
};

class Position {
public:
  Position();

  [[nodiscard]] static Position initial();

  [[nodiscard]] std::string toFEN() const;
  [[nodiscard]] static Position fromFEN(std::string_view fen);

  [[nodiscard]] Color turn() const { return turn_; }
  [[nodiscard]] bool bastionRight(Color c) const { return bastionRight_[static_cast<int>(c)]; }
  // New rule: a player may not build walls on two consecutive turns. This flag records
  // whether `c` built a wall on their previous turn (and is therefore blocked from building now).
  [[nodiscard]] bool wallBuiltLast(Color c) const { return wallBuiltLast_[static_cast<int>(c)]; }
  [[nodiscard]] int wallTokens(Color c) const { return wallTokens_[static_cast<int>(c)]; }
  [[nodiscard]] std::uint8_t sovereignSq(Color c) const { return sovereignSq_[static_cast<int>(c)]; }

  [[nodiscard]] uint64_t hash() const { return hash_; }
  [[nodiscard]] bool isRepetition() const;

  [[nodiscard]] bool gameOver() const { return winner_ != SQ_NONE; }
  [[nodiscard]] std::optional<Color> winner() const;
  [[nodiscard]] WinReason winReason() const { return winReason_; }

  // Move generation includes all turn-actions (move/capture, construct, command, demolish, bastion).
  void generateMoves(MoveList& out);

  void makeMove(const Move& m, Undo& u);
  void undoMove(const Undo& u);

  void makeNullMove(NullUndo& u);
  void undoNullMove(const NullUndo& u);

  [[nodiscard]] std::string pretty() const;

  // Low-level inspection (useful for eval / debugging).
  [[nodiscard]] std::int8_t rawAt(std::uint8_t s) const { return b_[s]; }
  [[nodiscard]] std::uint32_t pieceCount(Color c, PieceType pt) const {
    return pieceBB_[static_cast<std::size_t>(c)][static_cast<std::size_t>(pt)].popcount();
  }

  // Public for tooling/debug; core rules.
  [[nodiscard]] bool hasDominance(Color c) const;
  [[nodiscard]] bool isEntombed(Color victim) const;
  [[nodiscard]] Bitboard81 computeAttacks(Color attacker) const;

private:
  // Board encoding (signed):
  //  0  : empty
  //  ±1..±6 : pieces (abs(v)-1 => PieceType), sign => color
  //  ±7 : wall hp1 (abs(v)-6)
  //  ±8 : wall hp2
  std::array<std::int8_t, SQ_N> b_{};

  Color turn_ = Color::White;
  bool bastionRight_[2]{true, true};
  bool wallBuiltLast_[2]{false, false};
  std::uint8_t sovereignSq_[2]{SQ_NONE, SQ_NONE};
  int wallTokens_[2]{0, 0}; // sum of wall HP (reinforced counts as 2)
  int halfmove_ = 0;
  int fullmove_ = 1;
  std::uint8_t winner_ = SQ_NONE; // SQ_NONE = none, otherwise Color as 0/1
  WinReason winReason_ = WinReason::None;

  std::array<std::array<Bitboard81, static_cast<int>(PieceType::Count)>, 2> pieceBB_{};
  std::array<Bitboard81, 2> piecesBB_{};
  std::array<Bitboard81, 2> wallsBB_{};
  std::array<Bitboard81, 2> wallsReinfBB_{};

  uint64_t hash_ = 0;
  std::vector<uint64_t> history_;

  [[nodiscard]] static constexpr bool isPieceVal(std::int8_t v) {
    const int a = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
    return a >= 1 && a <= 6;
  }
  [[nodiscard]] static constexpr bool isWallVal(std::int8_t v) {
    const int a = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
    return a >= 7;
  }
  [[nodiscard]] static constexpr Color colorOf(std::int8_t v) { return (v > 0) ? Color::White : Color::Black; }
  [[nodiscard]] static constexpr PieceType pieceOf(std::int8_t v) {
    const int a = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
    return static_cast<PieceType>(a - 1);
  }
  [[nodiscard]] static constexpr int wallHp(std::int8_t v) {
    const int a = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
    return a - 6; // 1 or 2
  }
  [[nodiscard]] static constexpr std::int8_t makePiece(Color c, PieceType p) {
    const int n = 1 + static_cast<int>(p);
    return (c == Color::White) ? static_cast<std::int8_t>(n) : static_cast<std::int8_t>(-n);
  }
  [[nodiscard]] static constexpr std::int8_t makeWall(Color c, int hp) {
    const int n = 6 + hp; // 7 or 8
    return (c == Color::White) ? static_cast<std::int8_t>(n) : static_cast<std::int8_t>(-n);
  }

  [[nodiscard]] std::int8_t at(std::uint8_t s) const { return b_[s]; }
  void setSquareRaw(std::uint8_t s, std::int8_t v);
  void rebuildDerived();

  void saveSquare(Undo& u, std::uint8_t s);

  [[nodiscard]] bool threatened(std::uint8_t square, Color forColor) const;
  [[nodiscard]] bool isSquareAttackedBy(Color attacker, std::uint8_t square) const;
  [[nodiscard]] int masonMoveRange(std::uint8_t masonSq, Color c) const;
  [[nodiscard]] int ministerMoveRange(std::uint8_t ministerSq, Color c) const;
  [[nodiscard]] int sovereignMoveRange(std::uint8_t sovereignSq, Color c) const;

  void finalizeTurn();
  void hitWall(std::uint8_t wallSq, Color byColor);

  void genNormalMovesForPiece(MoveList& out, std::uint8_t fromSq, PieceType pt, Color us);
  void genMasonExtras(MoveList& out, std::uint8_t masonSq, Color us, const Bitboard81& enemyAttacks);
  void genCatapultExtras(MoveList& out, std::uint8_t catSq, Color us);
  void genBastion(MoveList& out, std::uint8_t sovSq, Color us);
};

} // namespace citadel

