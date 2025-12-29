#include "citadel/position.hpp"

#include <cassert>
#include <cctype>
#include <sstream>

#include "citadel/tables.hpp"

namespace citadel {

Position::Position() {
  b_.fill(0);
  turn_ = Color::White;
  bastionRight_[0] = true;
  bastionRight_[1] = true;
  wallTokens_[0] = 0;
  wallTokens_[1] = 0;
  sovereignSq_[0] = SQ_NONE;
  sovereignSq_[1] = SQ_NONE;
  halfmove_ = 0;
  fullmove_ = 1;
  winner_ = SQ_NONE;
  winReason_ = WinReason::None;
  rebuildDerived();
}

Position Position::initial() {
  Position p;
  p.b_.fill(0);
  p.turn_ = Color::White;
  p.bastionRight_[0] = true;
  p.bastionRight_[1] = true;
  p.halfmove_ = 0;
  p.fullmove_ = 1;
  p.winner_ = SQ_NONE;
  p.winReason_ = WinReason::None;

  constexpr PieceType back[N] = {
    PieceType::Catapult,
    PieceType::Lancer,
    PieceType::Pegasus,
    PieceType::Minister,
    PieceType::Sovereign,
    PieceType::Minister,
    PieceType::Pegasus,
    PieceType::Lancer,
    PieceType::Catapult,
  };

  for (int c = 0; c < N; ++c) {
    p.b_[sq(8, c)] = makePiece(Color::White, back[c]);
    p.b_[sq(7, c)] = makePiece(Color::White, PieceType::Mason);
    p.b_[sq(0, c)] = makePiece(Color::Black, back[c]);
    p.b_[sq(1, c)] = makePiece(Color::Black, PieceType::Mason);
  }

  p.rebuildDerived();
  return p;
}

std::optional<Color> Position::winner() const {
  if (winner_ == SQ_NONE) return std::nullopt;
  return static_cast<Color>(winner_);
}

bool Position::hasDominance(Color c) const {
  const std::uint8_t s = sovereignSq_[static_cast<int>(c)];
  return isKeepSq(s);
}

int Position::masonMoveRange(std::uint8_t masonSq, Color c) const {
  return (hasDominance(c) && isKeepSq(masonSq)) ? 2 : 1;
}

int Position::ministerMoveRange(std::uint8_t ministerSq, Color c) const {
  const bool dom = hasDominance(c);
  return 2 + ((dom && isKeepSq(ministerSq)) ? 1 : 0);
}

int Position::sovereignMoveRange(std::uint8_t sovSq, Color c) const {
  if (wallTokens(c) > 15) return 0;
  const bool dom = hasDominance(c);
  return 1 + ((dom && isKeepSq(sovSq)) ? 1 : 0);
}

bool Position::threatened(std::uint8_t square, Color forColor) const {
  return isSquareAttackedBy(other(forColor), square);
}

bool Position::isSquareAttackedBy(Color attacker, std::uint8_t square) const {
  if (square == SQ_NONE) return false;
  if (isWallVal(at(square))) return false; // no piece attacks walls

  const auto& T = tables();
  const int r = row(square);
  const int c = col(square);

  // Mason attacks (forward diagonals), cannot attack walls (handled above).
  {
    const int f = (attacker == Color::White) ? -1 : 1;
    const int mr = r - f; // mason row that would attack into r via +f
    if (mr >= 0 && mr < N) {
      const int c1 = c - 1;
      if (c1 >= 0) {
        const std::int8_t v = at(sq(mr, c1));
        if (isPieceVal(v) && colorOf(v) == attacker && pieceOf(v) == PieceType::Mason) return true;
      }
      const int c2 = c + 1;
      if (c2 < N) {
        const std::int8_t v = at(sq(mr, c2));
        if (isPieceVal(v) && colorOf(v) == attacker && pieceOf(v) == PieceType::Mason) return true;
      }
    }
  }

  // Pegasus attacks (knight), cannot land on walls (handled above).
  for (std::uint8_t i = 0; i < T.knightCount[square]; ++i) {
    const std::uint8_t from = T.knightTargets[square][i];
    const std::int8_t v = at(from);
    if (isPieceVal(v) && colorOf(v) == attacker && pieceOf(v) == PieceType::Pegasus) return true;
  }

  // Catapult attacks (rook rays), walls block.
  for (std::uint8_t dir = 0; dir < 4; ++dir) {
    const std::uint8_t len = T.rayLen[square][dir];
    for (std::uint8_t step = 0; step < len; ++step) {
      const std::uint8_t tsq = T.ray[square][dir][step];
      const std::int8_t v = at(tsq);
      if (isWallVal(v)) break;
      if (isPieceVal(v)) {
        if (colorOf(v) == attacker && pieceOf(v) == PieceType::Catapult) return true;
        break;
      }
    }
  }

  // Lancer attacks (bishop rays), may pass through friendly masons, walls block.
  for (std::uint8_t dir = 4; dir < 8; ++dir) {
    const std::uint8_t len = T.rayLen[square][dir];
    for (std::uint8_t step = 0; step < len; ++step) {
      const std::uint8_t tsq = T.ray[square][dir][step];
      const std::int8_t v = at(tsq);
      if (isWallVal(v)) break;
      if (isPieceVal(v)) {
        if (colorOf(v) == attacker) {
          const PieceType pt = pieceOf(v);
          if (pt == PieceType::Lancer) return true;
          if (pt == PieceType::Mason) continue; // pass through
        }
        break;
      }
    }
  }

  // Minister (<=2, or 3 with dominance in Keep) and Sovereign (<=1, or 2 with dominance in Keep; 0 if immobilized).
  for (std::uint8_t dir = 0; dir < 8; ++dir) {
    const std::uint8_t len = T.rayLen[square][dir];
    const int maxSteps = (len < 3) ? static_cast<int>(len) : 3; // minister max is 3
    for (int step = 0; step < maxSteps; ++step) {
      const std::uint8_t tsq = T.ray[square][dir][static_cast<std::uint8_t>(step)];
      const std::int8_t v = at(tsq);
      if (isWallVal(v)) break;
      if (isPieceVal(v)) {
        if (colorOf(v) == attacker) {
          const PieceType pt = pieceOf(v);
          const int dist = step + 1;
          if (pt == PieceType::Minister) {
            if (dist <= ministerMoveRange(tsq, attacker)) return true;
          } else if (pt == PieceType::Sovereign) {
            if (dist <= sovereignMoveRange(tsq, attacker)) return true;
          }
        }
        break;
      }
    }
  }

  return false;
}

bool Position::isEntombed(Color victim) const {
  const std::uint8_t k = sovereignSq(victim);
  if (k == SQ_NONE) return false;

  const auto& T = tables();
  for (std::uint8_t i = 0; i < T.kingCount[k]; ++i) {
    const std::uint8_t adj = T.kingTargets[k][i];
    const std::int8_t v = at(adj);
    if (!isWallVal(v)) return false;
  }

  // Board edges count as blocked (JS: out-of-bounds are ignored), so only in-bounds neighbors must be walls.
  return true;
}

Bitboard81 Position::computeAttacks(Color attacker) const {
  Bitboard81 attacked{};
  const Color us = attacker;
  const Color them = other(us);
  (void)them;

  const bool dom = hasDominance(us);
  const int wallCount = wallTokens(us);
  const auto& T = tables();

  // Mason attacks (forward diagonals), cannot attack walls.
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Mason)];
    const int f = (us == Color::White) ? -1 : 1;
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      const int r = row(s);
      const int c = col(s);
      for (const int dc : {-1, 1}) {
        const int rr = r + f;
        const int cc = c + dc;
        if (!inBounds(rr, cc)) continue;
        const std::uint8_t tsq = sq(rr, cc);
        if (isWallVal(at(tsq))) continue;
        attacked.set(tsq);
      }
    }
  }

  // Pegasus attacks (knight), cannot land on walls.
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Pegasus)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      for (std::uint8_t i = 0; i < T.knightCount[s]; ++i) {
        const std::uint8_t tsq = T.knightTargets[s][i];
        if (isWallVal(at(tsq))) continue;
        attacked.set(tsq);
      }
    }
  }

  // Catapult attacks (rook rays), walls block.
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Catapult)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      for (std::uint8_t dir = 0; dir < 4; ++dir) {
        const std::uint8_t len = T.rayLen[s][dir];
        for (std::uint8_t step = 0; step < len; ++step) {
          const std::uint8_t tsq = T.ray[s][dir][step];
          const std::int8_t v = at(tsq);
          if (isWallVal(v)) break;
          attacked.set(tsq);
          if (isPieceVal(v)) break;
        }
      }
    }
  }

  // Lancer attacks (bishop rays), may pass through friendly masons, walls block.
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Lancer)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      for (std::uint8_t dir = 4; dir < 8; ++dir) {
        const std::uint8_t len = T.rayLen[s][dir];
        for (std::uint8_t step = 0; step < len; ++step) {
          const std::uint8_t tsq = T.ray[s][dir][step];
          const std::int8_t v = at(tsq);
          if (isWallVal(v)) break;
          attacked.set(tsq);
          if (isPieceVal(v)) {
            if (colorOf(v) == us && pieceOf(v) == PieceType::Mason) continue; // pass through
            break;
          }
        }
      }
    }
  }

  // Minister attacks (up to 2, or 3 with dominance in Keep).
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Minister)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      const int max = 2 + ((dom && isKeepSq(s)) ? 1 : 0);
      for (std::uint8_t dir = 0; dir < 8; ++dir) {
        const std::uint8_t len = T.rayLen[s][dir];
        for (int step = 0; step < max && step < static_cast<int>(len); ++step) {
          const std::uint8_t tsq = T.ray[s][dir][static_cast<std::uint8_t>(step)];
          const std::int8_t v = at(tsq);
          if (isWallVal(v)) break;
          attacked.set(tsq);
          if (isPieceVal(v)) break;
        }
      }
    }
  }

  // Sovereign attacks (up to 1, or 2 with dominance in Keep). If immobilized (>15 wall tokens), no attacks.
  if (wallCount <= 15) {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Sovereign)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      const int max = 1 + ((dom && isKeepSq(s)) ? 1 : 0);
      for (std::uint8_t dir = 0; dir < 8; ++dir) {
        const std::uint8_t len = T.rayLen[s][dir];
        for (int step = 0; step < max && step < static_cast<int>(len); ++step) {
          const std::uint8_t tsq = T.ray[s][dir][static_cast<std::uint8_t>(step)];
          const std::int8_t v = at(tsq);
          if (isWallVal(v)) break;
          attacked.set(tsq);
          if (isPieceVal(v)) break;
        }
      }
    }
  }

  return attacked;
}

void Position::rebuildDerived() {
  for (int c = 0; c < 2; ++c) {
    piecesBB_[c] = Bitboard81{};
    wallsBB_[c] = Bitboard81{};
    wallsReinfBB_[c] = Bitboard81{};
    for (int p = 0; p < static_cast<int>(PieceType::Count); ++p) pieceBB_[c][p] = Bitboard81{};
  }
  wallTokens_[0] = 0;
  wallTokens_[1] = 0;
  sovereignSq_[0] = SQ_NONE;
  sovereignSq_[1] = SQ_NONE;

  const auto& T = tables();
  hash_ = 0;
  if (turn_ == Color::Black) hash_ ^= T.turnKey;
  for (int c = 0; c < 2; ++c) {
    if (bastionRight_[c]) hash_ ^= T.bastionKeys[c];
    if (wallBuiltLast_[c]) hash_ ^= T.wallBuiltLastKeys[c];
  }

  for (std::uint8_t s = 0; s < SQ_N; ++s) {
    const std::int8_t v = b_[s];
    if (v == 0) continue;
    if (isPieceVal(v)) {
      const Color c = colorOf(v);
      const PieceType pt = pieceOf(v);
      pieceBB_[static_cast<int>(c)][static_cast<int>(pt)].set(s);
      piecesBB_[static_cast<int>(c)].set(s);
      if (pt == PieceType::Sovereign) sovereignSq_[static_cast<int>(c)] = s;
      hash_ ^= T.pieceKeys[static_cast<int>(c)][static_cast<int>(pt)][s];
    } else if (isWallVal(v)) {
      const Color c = colorOf(v);
      wallsBB_[static_cast<int>(c)].set(s);
      const int hp = wallHp(v);
      wallTokens_[static_cast<int>(c)] += hp;
      if (hp == 2) wallsReinfBB_[static_cast<int>(c)].set(s);
      hash_ ^= T.wallKeys[static_cast<int>(c)][hp - 1][s];
    }
  }
}

bool Position::isRepetition() const {
  int count = 0;
  for (auto h : history_) {
    if (h == hash_) {
      if (++count >= 2) return true;
    }
  }
  return false;
}

void Position::setSquareRaw(std::uint8_t s, std::int8_t v) {
  const std::int8_t old = b_[s];
  if (old == v) return;

  const auto& T = tables();

  if (old != 0) {
    if (isPieceVal(old)) {
      const Color c = colorOf(old);
      const PieceType pt = pieceOf(old);
      pieceBB_[static_cast<int>(c)][static_cast<int>(pt)].reset(s);
      piecesBB_[static_cast<int>(c)].reset(s);
      hash_ ^= T.pieceKeys[static_cast<int>(c)][static_cast<int>(pt)][s];
    } else if (isWallVal(old)) {
      const Color c = colorOf(old);
      const int hp = wallHp(old);
      wallsBB_[static_cast<int>(c)].reset(s);
      if (hp == 2) wallsReinfBB_[static_cast<int>(c)].reset(s);
      hash_ ^= T.wallKeys[static_cast<int>(c)][hp - 1][s];
    }
  }

  b_[s] = v;

  if (v != 0) {
    if (isPieceVal(v)) {
      const Color c = colorOf(v);
      const PieceType pt = pieceOf(v);
      pieceBB_[static_cast<int>(c)][static_cast<int>(pt)].set(s);
      piecesBB_[static_cast<int>(c)].set(s);
      hash_ ^= T.pieceKeys[static_cast<int>(c)][static_cast<int>(pt)][s];
    } else if (isWallVal(v)) {
      const Color c = colorOf(v);
      const int hp = wallHp(v);
      wallsBB_[static_cast<int>(c)].set(s);
      if (hp == 2) wallsReinfBB_[static_cast<int>(c)].set(s);
      hash_ ^= T.wallKeys[static_cast<int>(c)][hp - 1][s];
    }
  }
}

void Position::saveSquare(Undo& u, std::uint8_t s) {
  for (std::uint8_t i = 0; i < u.sqCount; ++i) {
    if (u.sq[i] == s) return;
  }
  u.sq[u.sqCount] = s;
  u.prev[u.sqCount] = at(s);
  ++u.sqCount;
}

void Position::hitWall(std::uint8_t wallSq, Color /*byColor*/) {
  const std::int8_t v = at(wallSq);
  if (!isWallVal(v)) return;
  const Color owner = colorOf(v);
  const int hp = wallHp(v);
  if (hp == 2) {
    // 2 -> 1
    setSquareRaw(wallSq, makeWall(owner, 1));
    wallTokens_[static_cast<int>(owner)] -= 1;
  } else {
    // 1 -> 0
    setSquareRaw(wallSq, 0);
    wallTokens_[static_cast<int>(owner)] -= 1;
  }
}

void Position::finalizeTurn() {
  if (winner_ != SQ_NONE) return;

  const Color enemy = other(turn_);
  if (isEntombed(enemy)) {
    winner_ = static_cast<std::uint8_t>(turn_);
    winReason_ = WinReason::Entombment;
    halfmove_ = 0;
    return;
  }

  const auto& T = tables();
  hash_ ^= T.turnKey;

  const Color prev = turn_;
  turn_ = enemy;
  if (prev == Color::Black) ++fullmove_;
}

void Position::genNormalMovesForPiece(MoveList& out, std::uint8_t fromSq, PieceType pt, Color us) {
  const auto& T = tables();
  const Color them = other(us);
  const std::int8_t srcV = at(fromSq);
  (void)srcV;

  switch (pt) {
    case PieceType::Mason: {
      const int f = (us == Color::White) ? -1 : 1;
      const int max = masonMoveRange(fromSq, us);
      const int r = row(fromSq);
      const int c = col(fromSq);

      // Orthogonal (forward + sideways), empty only.
      const Coord ortho[3] = {{f, 0}, {0, -1}, {0, 1}};
      for (const auto& d : ortho) {
        for (int step = 1; step <= max; ++step) {
          const int rr = r + d.r * step;
          const int cc = c + d.c * step;
          if (!inBounds(rr, cc)) break;
          const std::uint8_t tsq = sq(rr, cc);
          if (at(tsq) != 0) break; // blocked by piece or wall
          out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
        }
      }

      // Diagonal captures (always 1).
      for (const int dc : {-1, 1}) {
        const int rr = r + f;
        const int cc = c + dc;
        if (!inBounds(rr, cc)) continue;
        const std::uint8_t tsq = sq(rr, cc);
        const std::int8_t v = at(tsq);
        if (isWallVal(v)) continue;
        if (isPieceVal(v) && colorOf(v) == them) out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
      }
      return;
    }

    case PieceType::Pegasus: {
      for (std::uint8_t i = 0; i < T.knightCount[fromSq]; ++i) {
        const std::uint8_t tsq = T.knightTargets[fromSq][i];
        const std::int8_t v = at(tsq);
        if (isWallVal(v)) continue;
        if (isPieceVal(v) && colorOf(v) == us) continue;
        out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
      }
      return;
    }

    case PieceType::Lancer: {
      for (std::uint8_t dir = 4; dir < 8; ++dir) {
        const std::uint8_t len = T.rayLen[fromSq][dir];
        for (std::uint8_t step = 0; step < len; ++step) {
          const std::uint8_t tsq = T.ray[fromSq][dir][step];
          const std::int8_t v = at(tsq);
          if (isWallVal(v)) break;
          if (isPieceVal(v)) {
            if (colorOf(v) == us && pieceOf(v) == PieceType::Mason) continue; // pass through
            if (colorOf(v) == them) out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
            break;
          }
          out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
        }
      }
      return;
    }

    case PieceType::Minister: {
      const int max = ministerMoveRange(fromSq, us);
      for (std::uint8_t dir = 0; dir < 8; ++dir) {
        const std::uint8_t len = T.rayLen[fromSq][dir];
        for (int step = 0; step < max && step < static_cast<int>(len); ++step) {
          const std::uint8_t tsq = T.ray[fromSq][dir][static_cast<std::uint8_t>(step)];
          const std::int8_t v = at(tsq);
          if (isWallVal(v)) break;
          if (isPieceVal(v)) {
            if (colorOf(v) == them) out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
            break;
          }
          out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
        }
      }
      return;
    }

    case PieceType::Sovereign: {
      const int max = sovereignMoveRange(fromSq, us);
      if (max <= 0) return;
      for (std::uint8_t dir = 0; dir < 8; ++dir) {
        const std::uint8_t len = T.rayLen[fromSq][dir];
        for (int step = 0; step < max && step < static_cast<int>(len); ++step) {
          const std::uint8_t tsq = T.ray[fromSq][dir][static_cast<std::uint8_t>(step)];
          const std::int8_t v = at(tsq);
          if (isWallVal(v)) break;
          if (isPieceVal(v)) {
            if (colorOf(v) == them) out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
            break;
          }
          out.push(Move{MoveType::Normal, fromSq, tsq, SQ_NONE, SQ_NONE});
        }
      }
      return;
    }

    case PieceType::Catapult:
    case PieceType::Count:
    default:
      return;
  }
}

void Position::genMasonExtras(MoveList& out, std::uint8_t masonSq, Color us, const Bitboard81& enemyAttacks) {
  const auto& T = tables();
  const Color them = other(us);
  const int r = row(masonSq);
  const int c = col(masonSq);
  const bool canBuild = !wallBuiltLast(us);

  // Construct: only if not threatened.
  if (canBuild && !enemyAttacks.test(masonSq)) {
    for (const auto& d : DIRS4) {
      const int rr = r + d.r;
      const int cc = c + d.c;
      if (!inBounds(rr, cc)) continue;
      const std::uint8_t tsq = sq(rr, cc);
      if (at(tsq) != 0) continue;
      out.push(Move{MoveType::MasonConstruct, masonSq, tsq, SQ_NONE, SQ_NONE});
    }
  }

  // Command: requires adjacent friendly minister.
  bool eligible = false;
  for (std::uint8_t i = 0; i < T.kingCount[masonSq]; ++i) {
    const std::uint8_t adj = T.kingTargets[masonSq][i];
    const std::int8_t v = at(adj);
    if (!isPieceVal(v) || colorOf(v) != us) continue;
    if (pieceOf(v) == PieceType::Minister) {
      eligible = true;
      break;
    }
  }
  if (!eligible) return;

  const int f = (us == Color::White) ? -1 : 1;
  const Coord ortho[3] = {{f, 0}, {0, -1}, {0, 1}};

  auto considerCommandDest = [&](std::uint8_t destSq) {
    const std::int8_t dstV = at(destSq);

    // If capturing the sovereign, command ends immediately (no build).
    if (isPieceVal(dstV) && colorOf(dstV) == them && pieceOf(dstV) == PieceType::Sovereign) {
      out.push(Move{MoveType::MasonCommand, masonSq, destSq, SQ_NONE, SQ_NONE});
      return;
    }

    const std::int8_t fromV = at(masonSq);
    const std::int8_t toV = dstV;

    // Apply the 1-step move (temporary) to evaluate threats / build squares.
    setSquareRaw(destSq, fromV);
    setSquareRaw(masonSq, 0);

    // Always allow skipping the build.
    out.push(Move{MoveType::MasonCommand, masonSq, destSq, SQ_NONE, SQ_NONE});

    if (canBuild && !isSquareAttackedBy(them, destSq)) {
      // Build targets: orth adjacent empties.
      const int nr = row(destSq);
      const int nc = col(destSq);
      for (const auto& d : DIRS4) {
        const int rr = nr + d.r;
        const int cc = nc + d.c;
        if (!inBounds(rr, cc)) continue;
        const std::uint8_t wsq = sq(rr, cc);
        if (at(wsq) == 0) out.push(Move{MoveType::MasonCommand, masonSq, destSq, wsq, SQ_NONE});
      }
    }

    // Restore
    setSquareRaw(masonSq, fromV);
    setSquareRaw(destSq, toV);
  };

  // Orth moves (empty only).
  for (const auto& d : ortho) {
    const int rr = r + d.r;
    const int cc = c + d.c;
    if (!inBounds(rr, cc)) continue;
    const std::uint8_t tsq = sq(rr, cc);
    if (at(tsq) == 0) considerCommandDest(tsq);
  }

  // Diagonal captures (enemy piece, not wall).
  for (const int dc2 : {-1, 1}) {
    const int rr = r + f;
    const int cc = c + dc2;
    if (!inBounds(rr, cc)) continue;
    const std::uint8_t tsq = sq(rr, cc);
    const std::int8_t v = at(tsq);
    if (isWallVal(v)) continue;
    if (isPieceVal(v) && colorOf(v) == them) considerCommandDest(tsq);
  }
}

void Position::genCatapultExtras(MoveList& out, std::uint8_t catSq, Color us) {
  const auto& T = tables();

  // Ranged demolish: first wall in any orthogonal ray (pieces block).
  for (std::uint8_t dir = 0; dir < 4; ++dir) {
    const std::uint8_t len = T.rayLen[catSq][dir];
    for (std::uint8_t step = 0; step < len; ++step) {
      const std::uint8_t tsq = T.ray[catSq][dir][step];
      const std::int8_t v = at(tsq);
      if (isPieceVal(v)) break;
      if (isWallVal(v)) {
        out.push(Move{MoveType::CatapultRangedDemolish, catSq, tsq, SQ_NONE, SQ_NONE});
        break;
      }
    }
  }

  // Normal rook moves/captures, then optional adjacent demolish.
  for (std::uint8_t dir = 0; dir < 4; ++dir) {
    const std::uint8_t len = T.rayLen[catSq][dir];
    for (std::uint8_t step = 0; step < len; ++step) {
      const std::uint8_t toSq = T.ray[catSq][dir][step];
      const std::int8_t dstV = at(toSq);
      if (isWallVal(dstV)) break;

      if (isPieceVal(dstV)) {
        if (colorOf(dstV) != us) {
          // Capture
          if (pieceOf(dstV) == PieceType::Sovereign) {
            out.push(Move{MoveType::CatapultMove, catSq, toSq, SQ_NONE, SQ_NONE});
          } else {
            // Optional adjacent demolish
            std::array<std::uint8_t, 8> adjWalls{};
            std::uint8_t adjCount = 0;
            for (std::uint8_t i = 0; i < T.kingCount[toSq]; ++i) {
              const std::uint8_t adj = T.kingTargets[toSq][i];
              if (isWallVal(at(adj))) adjWalls[adjCount++] = adj;
            }
            out.push(Move{MoveType::CatapultMove, catSq, toSq, SQ_NONE, SQ_NONE});
            for (std::uint8_t i = 0; i < adjCount; ++i) {
              out.push(Move{MoveType::CatapultMove, catSq, toSq, adjWalls[i], SQ_NONE});
            }
          }
        }
        break; // blocked by piece
      }

      // Empty square move
      std::array<std::uint8_t, 8> adjWalls{};
      std::uint8_t adjCount = 0;
      for (std::uint8_t i = 0; i < T.kingCount[toSq]; ++i) {
        const std::uint8_t adj = T.kingTargets[toSq][i];
        if (isWallVal(at(adj))) adjWalls[adjCount++] = adj;
      }
      if (adjCount == 0) {
        out.push(Move{MoveType::CatapultMove, catSq, toSq, SQ_NONE, SQ_NONE});
      } else {
        out.push(Move{MoveType::CatapultMove, catSq, toSq, SQ_NONE, SQ_NONE}); // skip demolish
        for (std::uint8_t i = 0; i < adjCount; ++i) {
          out.push(Move{MoveType::CatapultMove, catSq, toSq, adjWalls[i], SQ_NONE});
        }
      }
    }
  }
}

void Position::genBastion(MoveList& out, std::uint8_t sovSq, Color us) {
  if (wallBuiltLast(us)) return;
  if (!bastionRight(us)) return;
  if (wallTokens(us) > 15) return; // Siege Attrition disables Bastion (treated as movement).

  const auto& T = tables();
  for (std::uint8_t i = 0; i < T.kingCount[sovSq]; ++i) {
    const std::uint8_t ministerSq = T.kingTargets[sovSq][i];
    const std::int8_t v = at(ministerSq);
    if (!isPieceVal(v) || colorOf(v) != us || pieceOf(v) != PieceType::Minister) continue;

    // After swap, sovereign moves to ministerSq. Require >=2 empty adjacent squares (excluding the old sovereign square).
    std::array<std::uint8_t, 8> empties{};
    std::uint8_t ecount = 0;
    for (std::uint8_t j = 0; j < T.kingCount[ministerSq]; ++j) {
      const std::uint8_t adj = T.kingTargets[ministerSq][j];
      if (adj == sovSq) continue; // occupied by minister after swap
      if (at(adj) == 0) empties[ecount++] = adj;
    }
    if (ecount < 2) continue;

    for (std::uint8_t a = 0; a < ecount; ++a) {
      for (std::uint8_t b = static_cast<std::uint8_t>(a + 1); b < ecount; ++b) {
        out.push(Move{MoveType::Bastion, sovSq, ministerSq, empties[a], empties[b]});
      }
    }
  }
}

void Position::generateMoves(MoveList& out) {
  out.clear();
  if (gameOver()) return;

  const Color us = turn_;
  const Color them = other(us);
  const Bitboard81 enemyAttacks = computeAttacks(them);

  // Masons (+ construct + command)
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Mason)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      genNormalMovesForPiece(out, s, PieceType::Mason, us);
      genMasonExtras(out, s, us, enemyAttacks);
    }
  }

  // Pegasus
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Pegasus)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      genNormalMovesForPiece(out, s, PieceType::Pegasus, us);
    }
  }

  // Lancer
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Lancer)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      genNormalMovesForPiece(out, s, PieceType::Lancer, us);
    }
  }

  // Catapult (+ ranged demolish + optional adjacent demolish after move)
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Catapult)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      genCatapultExtras(out, s, us);
    }
  }

  // Minister
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Minister)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      genNormalMovesForPiece(out, s, PieceType::Minister, us);
    }
  }

  // Sovereign (+ Bastion)
  {
    Bitboard81 bb = pieceBB_[static_cast<int>(us)][static_cast<int>(PieceType::Sovereign)];
    while (bb.any()) {
      const std::uint8_t s = bb.pop_lsb();
      genNormalMovesForPiece(out, s, PieceType::Sovereign, us);
      genBastion(out, s, us);
    }
  }
}

void Position::makeMove(const Move& m, Undo& u) {
  history_.push_back(hash_);

  // Save globals.
  u.prevTurn = turn_;
  u.prevBastionRight[0] = bastionRight_[0];
  u.prevBastionRight[1] = bastionRight_[1];
  u.prevWallBuiltLast[0] = wallBuiltLast_[0];
  u.prevWallBuiltLast[1] = wallBuiltLast_[1];
  u.prevSovereignSq[0] = sovereignSq_[0];
  u.prevSovereignSq[1] = sovereignSq_[1];
  u.prevWallTokens[0] = wallTokens_[0];
  u.prevWallTokens[1] = wallTokens_[1];
  u.prevHalfmove = halfmove_;
  u.prevFullmove = fullmove_;
  u.prevWinner = winner_;
  u.prevWinReason = winReason_;
  u.sqCount = 0;

  if (gameOver()) return;

  const Color us = turn_;
  const Color them = other(us);

  auto setWallBuiltLast = [&](Color c, bool v) {
    const int idx = static_cast<int>(c);
    if (wallBuiltLast_[idx] == v) return;
    hash_ ^= tables().wallBuiltLastKeys[idx];
    wallBuiltLast_[idx] = v;
  };

  auto captureSovereign = [&](std::uint8_t toSq, std::int8_t srcV) {
    saveSquare(u, toSq);
    saveSquare(u, m.from);
    setSquareRaw(toSq, srcV);
    setSquareRaw(m.from, 0);

    // Update sovereign square/rights if capturing piece is sovereign.
    if (isPieceVal(srcV) && pieceOf(srcV) == PieceType::Sovereign) {
      sovereignSq_[static_cast<int>(us)] = toSq;
      if (bastionRight_[static_cast<int>(us)]) {
        hash_ ^= tables().bastionKeys[static_cast<int>(us)];
        bastionRight_[static_cast<int>(us)] = false;
      }
    }

    sovereignSq_[static_cast<int>(them)] = SQ_NONE;
    winner_ = static_cast<std::uint8_t>(us);
    winReason_ = WinReason::Regicide;
    halfmove_ = 0;
    setWallBuiltLast(us, false);
  };

  switch (m.type) {
    case MoveType::Normal: {
      const std::int8_t srcV = at(m.from);
      const std::int8_t dstV = at(m.to);
      const bool isCap = isPieceVal(dstV) && colorOf(dstV) == them;
      if (isCap && pieceOf(dstV) == PieceType::Sovereign) {
        captureSovereign(m.to, srcV);
        return;
      }

      saveSquare(u, m.from);
      saveSquare(u, m.to);
      setSquareRaw(m.to, srcV);
      setSquareRaw(m.from, 0);

      if (isPieceVal(srcV) && pieceOf(srcV) == PieceType::Sovereign) {
        sovereignSq_[static_cast<int>(us)] = m.to;
        if (bastionRight_[static_cast<int>(us)]) {
          hash_ ^= tables().bastionKeys[static_cast<int>(us)];
          bastionRight_[static_cast<int>(us)] = false;
        }
      }

      halfmove_ = isCap ? 0 : (halfmove_ + 1);
      setWallBuiltLast(us, false);
      finalizeTurn();
      return;
    }

    case MoveType::MasonConstruct: {
      // Place a wall adjacent to the mason (orth), only if target is empty.
      const std::int8_t masonV = at(m.from);
      (void)masonV;
      const int hp = isKeepSq(m.from) ? 2 : 1;

      saveSquare(u, m.to);
      setSquareRaw(m.to, makeWall(us, hp));
      wallTokens_[static_cast<int>(us)] += hp;

      halfmove_ = 0;
      setWallBuiltLast(us, true);
      finalizeTurn();
      return;
    }

    case MoveType::MasonCommand: {
      const std::int8_t srcV = at(m.from);
      const std::int8_t dstV = at(m.to);
      const bool isCap = isPieceVal(dstV) && colorOf(dstV) == them;
      if (isCap && pieceOf(dstV) == PieceType::Sovereign) {
        // Capturing the sovereign ends immediately (no build).
        captureSovereign(m.to, srcV);
        return;
      }

      saveSquare(u, m.from);
      saveSquare(u, m.to);
      setSquareRaw(m.to, srcV);
      setSquareRaw(m.from, 0);

      bool didWall = false;
      if (m.aux1 != SQ_NONE) {
        const int hp = isKeepSq(m.to) ? 2 : 1;
        saveSquare(u, m.aux1);
        setSquareRaw(m.aux1, makeWall(us, hp));
        wallTokens_[static_cast<int>(us)] += hp;
        didWall = true;
      }

      halfmove_ = (isCap || didWall) ? 0 : (halfmove_ + 1);
      setWallBuiltLast(us, didWall);
      finalizeTurn();
      return;
    }

    case MoveType::CatapultRangedDemolish: {
      saveSquare(u, m.to);
      hitWall(m.to, us);
      halfmove_ = 0;
      setWallBuiltLast(us, false);
      finalizeTurn();
      return;
    }

    case MoveType::CatapultMove: {
      const std::int8_t srcV = at(m.from);
      const std::int8_t dstV = at(m.to);
      const bool isCap = isPieceVal(dstV) && colorOf(dstV) == them;
      if (isCap && pieceOf(dstV) == PieceType::Sovereign) {
        captureSovereign(m.to, srcV);
        return;
      }

      saveSquare(u, m.from);
      saveSquare(u, m.to);
      setSquareRaw(m.to, srcV);
      setSquareRaw(m.from, 0);

      bool didDemo = false;
      if (m.aux1 != SQ_NONE) {
        saveSquare(u, m.aux1);
        hitWall(m.aux1, us);
        didDemo = true;
      }

      halfmove_ = (isCap || didDemo) ? 0 : (halfmove_ + 1);
      setWallBuiltLast(us, false);
      finalizeTurn();
      return;
    }

    case MoveType::Bastion: {
      // Swap sovereign and minister, then place 2 walls (HP1) adjacent to new sovereign.
      const std::int8_t sovV = at(m.from);
      const std::int8_t minV = at(m.to);

      saveSquare(u, m.from);
      saveSquare(u, m.to);
      setSquareRaw(m.to, sovV);
      setSquareRaw(m.from, minV);

      sovereignSq_[static_cast<int>(us)] = m.to;
      if (bastionRight_[static_cast<int>(us)]) {
        hash_ ^= tables().bastionKeys[static_cast<int>(us)];
        bastionRight_[static_cast<int>(us)] = false;
      }

      // Place two Bastion walls (always HP1).
      saveSquare(u, m.aux1);
      setSquareRaw(m.aux1, makeWall(us, 1));
      wallTokens_[static_cast<int>(us)] += 1;

      saveSquare(u, m.aux2);
      setSquareRaw(m.aux2, makeWall(us, 1));
      wallTokens_[static_cast<int>(us)] += 1;

      halfmove_ = 0;
      setWallBuiltLast(us, true);
      finalizeTurn();
      return;
    }

    default:
      return;
  }
}

void Position::undoMove(const Undo& u) {
  // Restore squares.
  for (std::uint8_t i = 0; i < u.sqCount; ++i) {
    setSquareRaw(u.sq[i], u.prev[i]);
  }

  // Restore globals.
  turn_ = u.prevTurn;
  bastionRight_[0] = u.prevBastionRight[0];
  bastionRight_[1] = u.prevBastionRight[1];
  wallBuiltLast_[0] = u.prevWallBuiltLast[0];
  wallBuiltLast_[1] = u.prevWallBuiltLast[1];
  sovereignSq_[0] = u.prevSovereignSq[0];
  sovereignSq_[1] = u.prevSovereignSq[1];
  wallTokens_[0] = u.prevWallTokens[0];
  wallTokens_[1] = u.prevWallTokens[1];
  halfmove_ = u.prevHalfmove;
  fullmove_ = u.prevFullmove;
  winner_ = u.prevWinner;
  winReason_ = u.prevWinReason;

  // Restore hash/history for repetition detection.
  // makeMove() pushes the pre-move hash onto history_; undo should restore it.
  if (!history_.empty()) {
    hash_ = history_.back();
    history_.pop_back();
  }
}

void Position::makeNullMove(NullUndo& u) {
  u.prevTurn = turn_;
  u.prevFullmove = fullmove_;
  if (gameOver()) return;

  const Color prev = turn_;
  // Null move flips side-to-move; update hash used by repetition detection.
  hash_ ^= tables().turnKey;
  turn_ = other(turn_);
  if (prev == Color::Black) ++fullmove_;
}

void Position::undoNullMove(const NullUndo& u) {
  if (u.prevTurn != turn_) hash_ ^= tables().turnKey;
  turn_ = u.prevTurn;
  fullmove_ = u.prevFullmove;
}

std::string Position::pretty() const {
  std::ostringstream oss;
  oss << "Turn: " << colorName(turn_) << "  "
      << "Bastion rights: " << (bastionRight(Color::White) ? "W" : "-") << (bastionRight(Color::Black) ? "b" : "-")
      << "  Walls: W=" << wallTokens(Color::White) << " B=" << wallTokens(Color::Black) << "\n";
  if (winner_ != SQ_NONE) {
    oss << "Winner: " << colorName(static_cast<Color>(winner_)) << " (" << ((winReason_ == WinReason::Regicide) ? "Regicide" : "Entombment")
        << ")\n";
  }

  oss << "   A B C D E F G H I\n";
  for (int r = 0; r < N; ++r) {
    oss << (N - r) << "  ";
    for (int c = 0; c < N; ++c) {
      const std::uint8_t s = sq(r, c);
      const std::int8_t v = at(s);
      char ch = '.';
      if (v != 0) {
        if (isPieceVal(v)) {
          const PieceType pt = pieceOf(v);
          switch (pt) {
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
      }

      oss << ch;
      if (c != N - 1) oss << ' ';
    }
    oss << "\n";
  }
  return oss.str();
}

} // namespace citadel

