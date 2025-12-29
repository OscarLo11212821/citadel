#include "citadel/search.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "citadel/nnue.hpp"
#include "citadel/tables.hpp"

namespace citadel {

// --------------------------------------------------------------------------------------
// Local board helpers (mirrors Position encoding helpers)
// --------------------------------------------------------------------------------------

static inline int abs8(std::int8_t v) { return (v < 0) ? -static_cast<int>(v) : static_cast<int>(v); }
static inline bool isPieceVal(std::int8_t v) {
  const int a = abs8(v);
  return a >= 1 && a <= 6;
}
static inline bool isWallVal(std::int8_t v) {
  const int a = abs8(v);
  return a >= 7;
}
static inline Color colorOf(std::int8_t v) { return (v > 0) ? Color::White : Color::Black; }
static inline PieceType pieceOf(std::int8_t v) { return static_cast<PieceType>(abs8(v) - 1); }

// --------------------------------------------------------------------------------------
// Evaluation
// --------------------------------------------------------------------------------------

static constexpr int INF = 1'000'000'000;
static constexpr int MATE = 100'000'000; // Checkmate in Citadel is called Regicide/Entombment, but we name it MATE here for consistency with chess.
static constexpr int MAX_PLY = 256;
static constexpr int QS_MAX_DEPTH = 4; // cap quiescence extensions to keep it fast

// Order matches Position encoding: 0..5 = Mason, Catapult, Lancer, Pegasus, Minister, Sovereign
static constexpr std::array<int, 6> PIECE_VALUE_MAT = {100, 550, 350, 400, 450, 0};      // Sovereign is priceless; treat as 0 in static material.
static constexpr std::array<int, 6> PIECE_VALUE_ORDER = {100, 550, 350, 400, 450, 100000}; // For move ordering (captures), sovereign capture must dominate.

static constexpr int DOMINANCE_BONUS = 25;             // lower than before; PST handles "gravity" toward the Keep.
static constexpr int WALL_BASE_VALUE_PER_HP = 2;       // keep minimal to avoid valuing useless corner walls.
static constexpr int WALL_ADJ_SOV_BONUS = 15;          // walls adjacent to own sovereign are valuable protection.
static constexpr int WALL_CHOKE_BONUS = 6;             // walls on the Keep boundary ring can be useful, but don't overvalue early.
static constexpr int MASON_MINISTER_SYNERGY = 20;      // mason gets much stronger if it can Command.
static constexpr int ENTOMB_PRESSURE_WEIGHT = 18;      // retained from prior eval (still important tactically).
static constexpr int SIEGE_ATTRITION_PENALTY = 200;     // immobilized sovereign penalty.

// Endgame / wall heuristics
static constexpr int WALLS_MANY_START = 12;             // total wall HP where games start to "lock"
static constexpr int WALLS_MANY_FULL = 25;              // total wall HP considered very locked
static constexpr int NO_CAT_DRAWISH_SCALE_MAX = 256;    // max % shrink (in /256) when no catapults and walls are high
static constexpr int CATAPULT_EDGE_BONUS_MAX = 150;     // bonus for having catapult edge in locked endgames

// Opening/midgame heuristics: discourage early sovereign adventures and wall spam.
static constexpr int MAX_NON_SOV_PIECES = 34;           // initial position: 17 non-sovereign pieces per side
static constexpr int BASTION_RIGHT_OPENING_BONUS = 80;  // keeping Bastion available early is valuable
static constexpr int KING_WANDER_PEN = 45;              // per square away from start, scaled by opening
static constexpr int KING_KEEP_EARLY_PEN = 140;         // sovereign in Keep too early is dangerous
static constexpr int KING_ATTACKED_PEN = 700;           // enemy attacks sovereign square (immediate regicide threat)
static constexpr int KING_RING_ATTACK_PEN = 55;         // per adjacent square attacked around sovereign
static constexpr int WALL_TOKEN_OPENING_PEN_PER_HP = 3; // discourage over-building walls early
static constexpr int MOBILITY_ATK_WEIGHT = 2;           // activity: reward attacked squares (proxy for mobility/development)

static constexpr int pstCentrality(int r, int c) {
  // Chebyshev distance from center (4,4) on a 9x9 board: 0..4
  const int dr = (r >= 4) ? (r - 4) : (4 - r);
  const int dc = (c >= 4) ? (c - 4) : (4 - c);
  const int cheb = (dr > dc) ? dr : dc;
  return 4 - cheb; // 4 at center, 0 on edge
}

static std::array<std::array<int, SQ_N>, static_cast<int>(PieceType::Count)> buildPST() {
  std::array<std::array<int, SQ_N>, static_cast<int>(PieceType::Count)> pst{};

  for (std::uint8_t s = 0; s < SQ_N; ++s) {
    const int r = row(s);
    const int c = col(s);
    const int cent = pstCentrality(r, c); // 0..4
    const bool keep = isKeep(r, c);
    const int keepBonus = keep ? 1 : 0;

    pst[static_cast<std::size_t>(PieceType::Mason)][s] = (cent * 4) + (keepBonus * 6);
    pst[static_cast<std::size_t>(PieceType::Catapult)][s] = (cent * 3) + (keepBonus * 4);
    pst[static_cast<std::size_t>(PieceType::Lancer)][s] = (cent * 4) + (keepBonus * 6);
    pst[static_cast<std::size_t>(PieceType::Pegasus)][s] = (cent * 4) + (keepBonus * 6);
    pst[static_cast<std::size_t>(PieceType::Minister)][s] = (cent * 5) + (keepBonus * 8);

    // Sovereign PST is intentionally much larger to create strong "gravity" toward the Keep.
    pst[static_cast<std::size_t>(PieceType::Sovereign)][s] = (cent * 20) + (keepBonus * 40);
  }

  return pst;
}

static const std::array<std::array<int, SQ_N>, static_cast<int>(PieceType::Count)>& PST() {
  static const auto pst = buildPST();
  return pst;
}

static constexpr bool isKeepBoundaryRing(int r, int c) {
  // A 5x5 "ring" around the Keep (Keep is 3..5). This corresponds to r/c in [2..6]
  // and on the boundary of that box. These squares are typical entry chokepoints.
  if (r < 2 || r > 6 || c < 2 || c > 6) return false;
  if (isKeep(r, c)) return false;
  return (r == 2 || r == 6 || c == 2 || c == 6);
}

static int evalStatic(const Position& pos) {
  // Positive = good for White.
  int scoreW = 0;
  int scoreB = 0;

  const auto& T = tables();

  // Game phase: 0 = opening, 256 = endgame (fewer pieces).
  int nonSovPieces = 0;
  for (std::uint8_t s = 0; s < SQ_N; ++s) {
    const std::int8_t v = pos.rawAt(s);
    if (v == 0) continue;
    const int av = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
    if (av >= 1 && av <= 6) {
      const int ptIdx = av - 1;
      if (ptIdx != static_cast<int>(PieceType::Sovereign)) ++nonSovPieces;
    }
  }
  int missing = MAX_NON_SOV_PIECES - nonSovPieces;
  if (missing < 0) missing = 0;
  const int phase = (missing * 256 + (MAX_NON_SOV_PIECES / 2)) / MAX_NON_SOV_PIECES; // 0..256
  const int opening = 256 - phase;

  const int wallsW = pos.wallTokens(Color::White);
  const int wallsB = pos.wallTokens(Color::Black);
  const int totalWalls = wallsW + wallsB;

  auto clamp256 = [](int x) -> int { return (x < 0) ? 0 : (x > 256) ? 256 : x; };
  const int wallMany = clamp256(((totalWalls - WALLS_MANY_START) * 256) / (WALLS_MANY_FULL - WALLS_MANY_START));
  const int wallEndgame = (wallMany * phase) / 256; // 0..256

  // 1. Calculate Sovereign Safety (Denominators for Proximity Heuristic)
  auto calculateSafety = [&](Color c) -> int {
    const std::uint8_t ks = pos.sovereignSq(c);
    if (ks == SQ_NONE) return 100; // King dead or missing? Treat as infinite safety to avoid div/0 errors, though game should be over.
    
    int safety = 1; // Base safety
    int wallSafetyCount = 0;
    
    for (std::uint8_t i = 0; i < T.kingCount[ks]; ++i) {
      const std::uint8_t adj = T.kingTargets[ks][i];
      const std::int8_t v = pos.rawAt(adj);
      if (v == 0) continue;
      
      // Friendly Piece: +2 safety (blocker + potential helper)
      if (isPieceVal(v) && colorOf(v) == c) {
        safety += 2;
      }
      // Friendly Wall: +1 safety (blocker), cap wall contribution at 3.
      // (Too many walls = entombment risk, not safety).
      else if (isWallVal(v) && colorOf(v) == c) {
        if (wallSafetyCount < 3) {
          safety += 1;
          wallSafetyCount++;
        }
      }
    }
    return safety;
  };

  const int safetyW = calculateSafety(Color::White);
  const int safetyB = calculateSafety(Color::Black);
  int pressureOnW = 0;
  int pressureOnB = 0;

  const std::uint8_t sovSqW = pos.sovereignSq(Color::White);
  const std::uint8_t sovSqB = pos.sovereignSq(Color::Black);

  auto dynPieceValue = [&](PieceType pt) -> int {
    const int idx = static_cast<int>(pt);
    const int base = PIECE_VALUE_MAT[static_cast<std::size_t>(idx)];
    int target = base;
    if (pt == PieceType::Mason) target = 225;
    if (pt == PieceType::Pegasus) target = 500;
    if (pt == PieceType::Catapult) target = 600;
    return base + ((target - base) * wallEndgame) / 256;
  };

  // Main Board Loop
  for (std::uint8_t s = 0; s < SQ_N; ++s) {
    const std::int8_t v = pos.rawAt(s);
    if (v == 0) continue;

    const bool isWhite = v > 0;
    const int av = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
    int& score = isWhite ? scoreW : scoreB;

    if (av >= 1 && av <= 6) {
      // It is a Piece
      const int ptIdx = av - 1;
      const PieceType pt = static_cast<PieceType>(ptIdx);

      // A. Material & PST
      score += dynPieceValue(pt);
      if (pt == PieceType::Sovereign) {
        score += (PST()[static_cast<std::size_t>(pt)][s] * phase) / 256;
      } else {
        score += PST()[static_cast<std::size_t>(pt)][s];
      }

      // B. Sovereign Proximity / Vulnerability Heuristic
      // If this piece is attacking the enemy sovereign, calculate pressure.
      const std::uint8_t targetSov = isWhite ? sovSqB : sovSqW;
      if (targetSov != SQ_NONE) {
        const int r = row(s), c = col(s);
        const int tr = row(targetSov), tc = col(targetSov);
        const int dr = (r > tr) ? (r - tr) : (tr - r);
        const int dc = (c > tc) ? (c - tc) : (tc - c);
        const int dist = (dr > dc) ? dr : dc; // Chebyshev distance

        if (dist <= 4) {
          int baseWeight = 0;
          switch (pt) {
            case PieceType::Mason:    baseWeight = 10; break;
            case PieceType::Pegasus:  baseWeight = 10; break;
            case PieceType::Catapult: baseWeight = 6;  break;
            case PieceType::Lancer:   baseWeight = 6;  break;
            case PieceType::Minister: baseWeight = 3;  break;
            default: break;
          }
          // Formula: Weight * (5 - Distance)
          // Dist 1: Weight*4. Dist 4: Weight*1.
          const int pVal = baseWeight * (5 - dist);
          if (isWhite) pressureOnB += pVal; else pressureOnW += pVal;
        }
      }

      // C. Minister-Mason synergy
      if (pt == PieceType::Mason) {
        for (std::uint8_t i = 0; i < T.kingCount[s]; ++i) {
          const std::uint8_t adj = T.kingTargets[s][i];
          const std::int8_t v2 = pos.rawAt(adj);
          if (v2 == 0) continue;
          const int av2 = (v2 < 0) ? -static_cast<int>(v2) : static_cast<int>(v2);
          if (av2 != (1 + static_cast<int>(PieceType::Minister))) continue;
          if ((v2 > 0) == isWhite) {
            score += MASON_MINISTER_SYNERGY;
            break;
          }
        }
      }
    } else {
      // It is a Wall
      const int hp = av - 6;
      score += WALL_BASE_VALUE_PER_HP * hp;

      const int r = row(s);
      const int c = col(s);
      if (isKeepBoundaryRing(r, c)) score += (WALL_CHOKE_BONUS * phase) / 256;
    }
  }

  // Apply Proximity Scores (Pressure / Safety)
  // We apply this to the attacker's score.
  scoreW += (pressureOnB * 4) / safetyB; // *4 is a scaling factor to bring it to CP range (safety usually 1..5)
  scoreB += (pressureOnW * 4) / safetyW;

  // Global Heuristics
  if (pos.hasDominance(Color::White)) scoreW += (DOMINANCE_BONUS * phase) / 256;
  if (pos.hasDominance(Color::Black)) scoreB += (DOMINANCE_BONUS * phase) / 256;

  if (pos.bastionRight(Color::White)) scoreW += (BASTION_RIGHT_OPENING_BONUS * opening) / 256;
  if (pos.bastionRight(Color::Black)) scoreB += (BASTION_RIGHT_OPENING_BONUS * opening) / 256;

  // Helper for wall adjacency bonus
  auto addAdjWallBonus = [&](Color c, int& sc) {
    const std::uint8_t ks = pos.sovereignSq(c);
    if (ks == SQ_NONE) return;
    for (std::uint8_t i = 0; i < T.kingCount[ks]; ++i) {
      const std::uint8_t adj = T.kingTargets[ks][i];
      const std::int8_t v = pos.rawAt(adj);
      if (v == 0) continue;
      const int av = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
      if (av < 7) continue;
      const bool wallIsWhite = v > 0;
      if ((c == Color::White) == wallIsWhite) sc += WALL_ADJ_SOV_BONUS;
    }
  };
  addAdjWallBonus(Color::White, scoreW);
  addAdjWallBonus(Color::Black, scoreB);

  // Penalties
  if (pos.wallTokens(Color::White) > 15) scoreW -= SIEGE_ATTRITION_PENALTY;
  if (pos.wallTokens(Color::Black) > 15) scoreB -= SIEGE_ATTRITION_PENALTY;

  scoreW -= (pos.wallTokens(Color::White) * WALL_TOKEN_OPENING_PEN_PER_HP * opening) / 256;
  scoreB -= (pos.wallTokens(Color::Black) * WALL_TOKEN_OPENING_PEN_PER_HP * opening) / 256;

  // Mobility
  const Bitboard81 attW = pos.computeAttacks(Color::White);
  const Bitboard81 attB = pos.computeAttacks(Color::Black);
  const int mobW = static_cast<int>(attW.popcount());
  const int mobB = static_cast<int>(attB.popcount());
  scoreW += MOBILITY_ATK_WEIGHT * mobW;
  scoreB += MOBILITY_ATK_WEIGHT * mobB;

  // King Safety (Penalty-based)
  auto kingSafetyPen = [&](Color c, const Bitboard81& enemyAttacks) -> int {
    const std::uint8_t ks = pos.sovereignSq(c);
    if (ks == SQ_NONE) return 0;
    int pen = 0;
    const std::uint8_t home = (c == Color::White) ? sq(8, 4) : sq(0, 4);
    const int dr = std::abs(row(ks) - row(home));
    const int dc = std::abs(col(ks) - col(home));
    const int cheb = (dr > dc) ? dr : dc;
    pen += (KING_WANDER_PEN * cheb * opening) / 256;
    if (isKeepSq(ks)) pen += (KING_KEEP_EARLY_PEN * opening) / 256;
    if (enemyAttacks.test(ks)) pen += KING_ATTACKED_PEN;
    int ringAtt = 0;
    const int r0 = row(ks);
    const int c0 = col(ks);
    for (const auto& d : DIRS8) {
      const int rr = r0 + d.r;
      const int cc = c0 + d.c;
      if (!inBounds(rr, cc)) continue;
      const std::uint8_t adj = sq(rr, cc);
      if (enemyAttacks.test(adj)) ++ringAtt;
    }
    pen += KING_RING_ATTACK_PEN * ringAtt;
    return pen;
  };
  scoreW -= kingSafetyPen(Color::White, attB);
  scoreB -= kingSafetyPen(Color::Black, attW);

  // Entombment Pressure
  auto entombPressure = [&](Color attacker) -> int {
    const Color victim = other(attacker);
    const std::uint8_t vk = pos.sovereignSq(victim);
    if (vk == SQ_NONE) return 0;
    int blocked = 0;
    const int r0 = row(vk);
    const int c0 = col(vk);
    for (const auto& d : DIRS8) {
      const int rr = r0 + d.r;
      const int cc = c0 + d.c;
      if (!inBounds(rr, cc)) { ++blocked; continue; }
      const std::uint8_t adj = sq(rr, cc);
      const std::int8_t v = pos.rawAt(adj);
      const int av = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
      if (av >= 7) ++blocked;
    }
    return blocked;
  };
  scoreW += ENTOMB_PRESSURE_WEIGHT * entombPressure(Color::White);
  scoreB += ENTOMB_PRESSURE_WEIGHT * entombPressure(Color::Black);

  // ----------------------------------------------------------------------------
  // Tempo
  // ----------------------------------------------------------------------------
  // We apply Tempo here so that it gets scaled down if the position is deemed
  // drawish/locked later. This prevents score oscillation (+20/-20) in dead draws
  // while preserving initiative scores in open positions.
  constexpr int TEMPO_BONUS = 20;
  if (pos.turn() == Color::White) scoreW += TEMPO_BONUS;
  else scoreB += TEMPO_BONUS;

  int diff = scoreW - scoreB;

  // ----------------------------------------------------------------------------
  // Catapult / Wall Endgame & Draw Heuristics
  // ----------------------------------------------------------------------------
  const int catW = static_cast<int>(pos.pieceCount(Color::White, PieceType::Catapult));
  const int catB = static_cast<int>(pos.pieceCount(Color::Black, PieceType::Catapult));
  
  // Constant for Catapult Monopoly (one side has it, other doesn't)
  constexpr int CATAPULT_MONOPOLY_BONUS = 200;

  if (catW == 0 && catB == 0) {
    // 1. Both sides have NO Catapults. Walls are permanent.
    const int mobTotal = mobW + mobB;
    int drawish = clamp256(((60 - mobTotal) * 256) / 40);

    const int masons = static_cast<int>(pos.pieceCount(Color::White, PieceType::Mason) + 
                                        pos.pieceCount(Color::Black, PieceType::Mason));
    if (masons > 0) {
      // Masons present + No Catapults = Infinite Wall potential => High Draw Probability.
      // Apply strict score dampening.
      int masonFactor = 200; // ~78% score reduction
      if (totalWalls >= 4) masonFactor = 245; // ~95% score reduction
      drawish = std::max(drawish, masonFactor);
    } else {
      // No Masons, No Catapults. Static board.
      // If walls are high, it's likely drawn/locked.
      int staticWallFactor = (totalWalls * 20); 
      if (staticWallFactor > 256) staticWallFactor = 256;
      drawish = std::max(drawish, staticWallFactor);
    }

    const int scale = 256 - (drawish * NO_CAT_DRAWISH_SCALE_MAX) / 256;
    diff = (diff * scale) / 256;

  } else {
    // 2. At least one side has a Catapult.
    
    // Check for "Monopoly": One side has catapults, the other has NONE.
    // This is a massive strategic advantage (conversion potential) regardless of phase.
    if (catW > 0 && catB == 0) diff += CATAPULT_MONOPOLY_BONUS;
    else if (catB > 0 && catW == 0) diff -= CATAPULT_MONOPOLY_BONUS;

    // Small edge bonus for having *more* catapults in endgame (e.g. 2 vs 1)
    if (catW != catB) {
      const int edge = (catW > catB) ? 1 : -1;
      const int bonus = (CATAPULT_EDGE_BONUS_MAX * wallEndgame) / 256;
      diff += edge * bonus;
    }
  }

  return diff;
}

static inline int hceEvalStm(const Position& pos) {
  const int diff = evalStatic(pos);
  return (pos.turn() == Color::White) ? diff : -diff;
}

struct ScoredMove {
  Move m;
  int score = 0;
};

static int moveHeuristic(const Position& pos, const Move& m) {
  // Cheap move ordering: captures > wall hits/builds > quiet.
  int sc = 0;
  const Color us = pos.turn();
  const Color them = other(us);
  (void)them;

  auto av = [&](std::int8_t v) { return (v < 0) ? -static_cast<int>(v) : static_cast<int>(v); };

  switch (m.type) {
    case MoveType::Normal:
    case MoveType::CatapultMove:
    case MoveType::MasonCommand: {
      const std::int8_t dstV = pos.rawAt(m.to);
      const int a = av(dstV);
      if (a >= 1 && a <= 6) sc += 10'000 + PIECE_VALUE_ORDER[static_cast<std::size_t>(a - 1)];
      break;
    }
    case MoveType::CatapultRangedDemolish:
      sc += 8'000;
      break;
    case MoveType::MasonConstruct:
      sc += 6'000;
      break;
      break;
    default:
      break;
  }

  if ((m.type == MoveType::CatapultMove && m.aux1 != SQ_NONE) || (m.type == MoveType::MasonCommand && m.aux1 != SQ_NONE))
    sc += 1'000;

  return sc;
}

// --------------------------------------------------------------------------------------
// Zobrist hashing + Transposition Table
// --------------------------------------------------------------------------------------

static inline std::uint64_t splitmix64(std::uint64_t& x) {
  std::uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

struct ZobristKeys {
  // [square][pieceIndex], where pieceIndex in [0..15]:
  //   0..5  white pieces, 6..7 white walls (hp1/hp2)
  //   8..13 black pieces, 14..15 black walls (hp1/hp2)
  std::array<std::array<std::uint64_t, 16>, SQ_N> sq{};
  std::uint64_t turn = 0; // black to move
  std::uint64_t bastionW = 0;
  std::uint64_t bastionB = 0;
  std::uint64_t wallBuiltW = 0;
  std::uint64_t wallBuiltB = 0;
};

static ZobristKeys buildZobrist() {
  ZobristKeys k{};
  std::uint64_t seed = 0xC1ADEC1ULL; // deterministic seed
  for (std::uint8_t s = 0; s < SQ_N; ++s) {
    for (int i = 0; i < 16; ++i) k.sq[s][static_cast<std::size_t>(i)] = splitmix64(seed);
  }
  k.turn = splitmix64(seed);
  k.bastionW = splitmix64(seed);
  k.bastionB = splitmix64(seed);
  k.wallBuiltW = splitmix64(seed);
  k.wallBuiltB = splitmix64(seed);
  return k;
}

static const ZobristKeys& zobrist() {
  static const ZobristKeys k = buildZobrist();
  return k;
}

static inline int zobristIndex(std::int8_t v) {
  // v != 0. Maps Position encoding to [0..15] per ZobristKeys description.
  const bool isWhite = v > 0;
  const int av = (v < 0) ? -static_cast<int>(v) : static_cast<int>(v);
  const int base = isWhite ? 0 : 8;
  if (av >= 1 && av <= 6) return base + (av - 1);
  // Wall: 7->hp1, 8->hp2
  return base + 6 + (av - 7);
}

static std::uint64_t hashPosition(const Position& pos) {
  const auto& Z = zobrist();
  std::uint64_t h = 0;
  for (std::uint8_t s = 0; s < SQ_N; ++s) {
    const std::int8_t v = pos.rawAt(s);
    if (v == 0) continue;
    h ^= Z.sq[s][static_cast<std::size_t>(zobristIndex(v))];
  }
  if (pos.turn() == Color::Black) h ^= Z.turn;
  if (pos.bastionRight(Color::White)) h ^= Z.bastionW;
  if (pos.bastionRight(Color::Black)) h ^= Z.bastionB;
  if (pos.wallBuiltLast(Color::White)) h ^= Z.wallBuiltW;
  if (pos.wallBuiltLast(Color::Black)) h ^= Z.wallBuiltB;
  return h;
}

static std::uint64_t hashAfterMake(std::uint64_t h, const Position& pos, const Undo& u) {
  const auto& Z = zobrist();

  for (std::uint8_t i = 0; i < u.sqCount; ++i) {
    const std::uint8_t s = u.sq[i];
    const std::int8_t oldV = u.prev[i];
    const std::int8_t newV = pos.rawAt(s);
    if (oldV != 0) h ^= Z.sq[s][static_cast<std::size_t>(zobristIndex(oldV))];
    if (newV != 0) h ^= Z.sq[s][static_cast<std::size_t>(zobristIndex(newV))];
  }

  if (u.prevTurn != pos.turn()) h ^= Z.turn;
  if (u.prevBastionRight[static_cast<int>(Color::White)] != pos.bastionRight(Color::White)) h ^= Z.bastionW;
  if (u.prevBastionRight[static_cast<int>(Color::Black)] != pos.bastionRight(Color::Black)) h ^= Z.bastionB;
  if (u.prevWallBuiltLast[static_cast<int>(Color::White)] != pos.wallBuiltLast(Color::White)) h ^= Z.wallBuiltW;
  if (u.prevWallBuiltLast[static_cast<int>(Color::Black)] != pos.wallBuiltLast(Color::Black)) h ^= Z.wallBuiltB;

  return h;
}

enum class TTFlag : std::uint8_t { Exact = 0, Lower = 1, Upper = 2 };

struct TTEntry {
  std::uint64_t key = 0;
  int score = 0;   // stored with mate-distance normalization (see scoreToTT/scoreFromTT)
  int depth = 0;   // remaining depth at this node
  TTFlag flag = TTFlag::Exact;
  Move best = nullMove();
};

static std::vector<TTEntry> TT{};
static std::size_t TT_MASK = 0;
static std::size_t TT_MB = 16;

static void allocTTMB(std::size_t mb) {
  if (mb < 1) mb = 1;
  if (mb > 1024) mb = 1024;
  TT_MB = mb;

  const std::size_t bytes = mb * 1024ull * 1024ull;
  std::size_t entries = bytes / sizeof(TTEntry);
  if (entries < 1024) entries = 1024;
  entries = std::bit_ceil(entries);

  TT.assign(entries, TTEntry{});
  TT_MASK = entries - 1;
}

static void ensureTT() {
  if (!TT.empty()) return;
  allocTTMB(TT_MB);
}

void clearTranspositionTable() {
  ensureTT();
  for (auto& e : TT) e = TTEntry{};
}

void setTranspositionTableSizeMB(std::size_t mb) {
  allocTTMB(mb);
}

std::size_t transpositionTableSizeMB() {
  ensureTT();
  return TT_MB;
}

static inline bool sameMove(const Move& a, const Move& b) {
  return a.type == b.type && a.from == b.from && a.to == b.to && a.aux1 == b.aux1 && a.aux2 == b.aux2;
}

static inline int scoreToTT(int score, int ply) {
  // Normalize mate scores so they stay consistent when retrieved at different ply.
  if (score > MATE - 10'000) return score + ply;
  if (score < -MATE + 10'000) return score - ply;
  return score;
}

static inline int scoreFromTT(int score, int ply) {
  if (score > MATE - 10'000) return score - ply;
  if (score < -MATE + 10'000) return score + ply;
  return score;
}

static inline TTEntry& ttSlot(std::uint64_t key) {
  return TT[static_cast<std::size_t>(key) & TT_MASK];
}

// --------------------------------------------------------------------------------------
// Quiescence + PVS Negamax
// --------------------------------------------------------------------------------------

static void generateNoisyMoves(const Position& pos, MoveList& out) {
  out.clear();
  if (pos.gameOver()) return;

  const auto& T = tables();
  const Color us = pos.turn();
  const Color them = other(us);
  const bool dom = pos.hasDominance(us);

  // Precompute squares adjacent to enemy sovereign (8-neighborhood).
  std::array<std::uint8_t, SQ_N> adjEnemy{};
  adjEnemy.fill(0);
  const std::uint8_t enemySov = pos.sovereignSq(them);
  if (enemySov != SQ_NONE) {
    for (std::uint8_t i = 0; i < T.kingCount[enemySov]; ++i) adjEnemy[T.kingTargets[enemySov][i]] = 1;
  }
  // Lazily computed: only needed for MasonConstruct (construct is illegal if the mason is threatened).
  bool haveEnemyAttacks = false;
  Bitboard81 enemyAttacks{};

  auto isEnemyPiece = [&](std::int8_t v) {
    return isPieceVal(v) && colorOf(v) == them;
  };
  auto isFriendlyPiece = [&](std::int8_t v) {
    return isPieceVal(v) && colorOf(v) == us;
  };

  for (std::uint8_t from = 0; from < SQ_N; ++from) {
    const std::int8_t srcV = pos.rawAt(from);
    if (!isFriendlyPiece(srcV)) continue;

    const PieceType pt = pieceOf(srcV);
    const int r = row(from);
    const int c = col(from);

    switch (pt) {
      case PieceType::Mason: {
        const int f = (us == Color::White) ? -1 : 1;
        // Diagonal captures (always 1).
        for (const int dc : {-1, 1}) {
          const int rr = r + f;
          const int cc = c + dc;
          if (!inBounds(rr, cc)) continue;
          const std::uint8_t to = sq(rr, cc);
          const std::int8_t dstV = pos.rawAt(to);
          if (isWallVal(dstV)) continue;
          if (isEnemyPiece(dstV)) out.push(Move{MoveType::Normal, from, to, SQ_NONE, SQ_NONE});
        }

        // "Noisy" wall construction near enemy sovereign (rare): only consider walls placed adjacent to enemy sovereign.
        if (enemySov != SQ_NONE) {
          for (const auto& d : DIRS4) {
            const int rr = r + d.r;
            const int cc = c + d.c;
            if (!inBounds(rr, cc)) continue;
            const std::uint8_t to = sq(rr, cc);
            if (!adjEnemy[to]) continue;
            if (pos.rawAt(to) != 0) continue;

            if (!haveEnemyAttacks) {
              enemyAttacks = pos.computeAttacks(them);
              haveEnemyAttacks = true;
            }
            if (!enemyAttacks.test(from)) out.push(Move{MoveType::MasonConstruct, from, to, SQ_NONE, SQ_NONE});
          }
        }
        break;
      }

      case PieceType::Pegasus: {
        for (std::uint8_t i = 0; i < T.knightCount[from]; ++i) {
          const std::uint8_t to = T.knightTargets[from][i];
          const std::int8_t dstV = pos.rawAt(to);
          if (isWallVal(dstV)) continue;
          if (isEnemyPiece(dstV)) out.push(Move{MoveType::Normal, from, to, SQ_NONE, SQ_NONE});
        }
        break;
      }

      case PieceType::Lancer: {
        for (std::uint8_t dir = 4; dir < 8; ++dir) {
          const std::uint8_t len = T.rayLen[from][dir];
          for (std::uint8_t step = 0; step < len; ++step) {
            const std::uint8_t to = T.ray[from][dir][step];
            const std::int8_t dstV = pos.rawAt(to);
            if (isWallVal(dstV)) break;
            if (isPieceVal(dstV)) {
              if (isFriendlyPiece(dstV) && pieceOf(dstV) == PieceType::Mason) continue; // pass through
              if (isEnemyPiece(dstV)) out.push(Move{MoveType::Normal, from, to, SQ_NONE, SQ_NONE});
              break;
            }
          }
        }
        break;
      }

      case PieceType::Minister: {
        const int max = 2 + ((dom && isKeepSq(from)) ? 1 : 0);
        for (std::uint8_t dir = 0; dir < 8; ++dir) {
          const std::uint8_t len = T.rayLen[from][dir];
          for (int step = 0; step < max && step < static_cast<int>(len); ++step) {
            const std::uint8_t to = T.ray[from][dir][static_cast<std::uint8_t>(step)];
            const std::int8_t dstV = pos.rawAt(to);
            if (isWallVal(dstV)) break;
            if (isPieceVal(dstV)) {
              if (isEnemyPiece(dstV)) out.push(Move{MoveType::Normal, from, to, SQ_NONE, SQ_NONE});
              break;
            }
          }
        }
        break;
      }

      case PieceType::Sovereign: {
        const int max = (pos.wallTokens(us) > 15) ? 0 : (1 + ((dom && isKeepSq(from)) ? 1 : 0));
        if (max <= 0) break;
        for (std::uint8_t dir = 0; dir < 8; ++dir) {
          const std::uint8_t len = T.rayLen[from][dir];
          for (int step = 0; step < max && step < static_cast<int>(len); ++step) {
            const std::uint8_t to = T.ray[from][dir][static_cast<std::uint8_t>(step)];
            const std::int8_t dstV = pos.rawAt(to);
            if (isWallVal(dstV)) break;
            if (isPieceVal(dstV)) {
              if (isEnemyPiece(dstV)) out.push(Move{MoveType::Normal, from, to, SQ_NONE, SQ_NONE});
              break;
            }
            // Sovereign quiet moves are only "noisy" if they interact with Keep geometry.
            if (isKeepSq(from) || isKeepSq(to)) out.push(Move{MoveType::Normal, from, to, SQ_NONE, SQ_NONE});
          }
        }
        break;
      }

      case PieceType::Catapult: {
        // Ranged demolish: first wall in any orthogonal ray (pieces block).
        for (std::uint8_t dir = 0; dir < 4; ++dir) {
          const std::uint8_t len = T.rayLen[from][dir];
          for (std::uint8_t step = 0; step < len; ++step) {
            const std::uint8_t to = T.ray[from][dir][step];
            const std::int8_t dstV = pos.rawAt(to);
            if (isPieceVal(dstV)) break;
            if (isWallVal(dstV)) {
              out.push(Move{MoveType::CatapultRangedDemolish, from, to, SQ_NONE, SQ_NONE});
              break;
            }
          }
        }

        // Captures and adjacent-demolish moves.
        for (std::uint8_t dir = 0; dir < 4; ++dir) {
          const std::uint8_t len = T.rayLen[from][dir];
          for (std::uint8_t step = 0; step < len; ++step) {
            const std::uint8_t to = T.ray[from][dir][step];
            const std::int8_t dstV = pos.rawAt(to);
            if (isWallVal(dstV)) break;

            if (isPieceVal(dstV)) {
              if (isEnemyPiece(dstV)) {
                out.push(Move{MoveType::CatapultMove, from, to, SQ_NONE, SQ_NONE}); // capture
                // Optional adjacent demolish (noisy) after capture.
                for (std::uint8_t i = 0; i < T.kingCount[to]; ++i) {
                  const std::uint8_t adj = T.kingTargets[to][i];
                  if (isWallVal(pos.rawAt(adj))) out.push(Move{MoveType::CatapultMove, from, to, adj, SQ_NONE});
                }
              }
              break;
            }

            // Empty-square move is only noisy if we also demolish a wall adjacent to the destination.
            for (std::uint8_t i = 0; i < T.kingCount[to]; ++i) {
              const std::uint8_t adj = T.kingTargets[to][i];
              if (isWallVal(pos.rawAt(adj))) out.push(Move{MoveType::CatapultMove, from, to, adj, SQ_NONE});
            }
          }
        }
        break;
      }

      default:
        break;
    }
  }
}

static inline int mateScore(int ply) {
  return MATE - ply;
}

static constexpr int MOVE_TYPE_N = 1 + static_cast<int>(MoveType::Bastion);
static constexpr std::size_t HISTORY_SIZE = static_cast<std::size_t>(MOVE_TYPE_N) * SQ_N * SQ_N;

static inline int nonSovPieceCount(const Position& pos, Color c) {
  return static_cast<int>(pos.pieceCount(c, PieceType::Mason) + pos.pieceCount(c, PieceType::Catapult) + pos.pieceCount(c, PieceType::Lancer) +
                          pos.pieceCount(c, PieceType::Pegasus) + pos.pieceCount(c, PieceType::Minister));
}

static inline bool isQuietMove(const Position& pos, const Move& m) {
  return m.type == MoveType::Normal && m.to != SQ_NONE && pos.rawAt(m.to) == 0;
}

static inline std::size_t historyIndex(const Move& m) {
  return (static_cast<std::size_t>(m.type) * SQ_N + static_cast<std::size_t>(m.from)) * SQ_N + static_cast<std::size_t>(m.to);
}

// NOTE: std::thread stacks are relatively small on macOS by default. Our MoveList (4096 moves)
// is large, so we avoid allocating it (and a parallel 4096-int score array) on the recursion stack.
struct PlyBuffers {
  std::vector<MoveList> moves;
  std::vector<std::array<int, 4096>> scores;
  std::vector<NNUE::Accumulator> nnueAcc;

  PlyBuffers() : moves(MAX_PLY), scores(MAX_PLY), nnueAcc(MAX_PLY) {}
};

static thread_local PlyBuffers PLY{};

struct SearchContext {
  SearchLimits limits{};
  SearchInfoCallback onInfo{};
  std::atomic_bool* stop = nullptr;

  EvalBackend evalBackend = EvalBackend::HCE;
  const NNUE* nnue = nullptr;
  bool useNNUE = false;
  bool useTT = true;

  std::chrono::steady_clock::time_point start{};
  std::chrono::steady_clock::time_point end{};
  bool useTime = false;
  std::uint64_t nodeLimit = 0;

  std::uint64_t nodes = 0;
  int seldepth = 0;
  bool aborted = false;

  std::array<std::array<Move, 2>, MAX_PLY> killers{};
  std::array<int, HISTORY_SIZE> history{};

  void resetHeuristics() {
    for (auto& k : killers) {
      k[0] = nullMove();
      k[1] = nullMove();
    }
    history.fill(0);
  }

  [[nodiscard]] std::uint64_t elapsedMs() const {
    using namespace std::chrono;
    return static_cast<std::uint64_t>(duration_cast<milliseconds>(steady_clock::now() - start).count());
  }

  bool shouldStop() {
    if (aborted) return true;
    // Check stop/time/node limits only every ~2k nodes (cheap + responsive enough for UCI).
    if ((nodes & 2047ull) != 0) return false;
    if (stop && stop->load(std::memory_order_relaxed)) {
      aborted = true;
      return true;
    }
    if (nodeLimit != 0 && nodes >= nodeLimit) {
      aborted = true;
      if (stop) stop->store(true, std::memory_order_relaxed);
      return true;
    }
    if (useTime && std::chrono::steady_clock::now() >= end) {
      aborted = true;
      if (stop) stop->store(true, std::memory_order_relaxed);
      return true;
    }
    return false;
  }
};

static inline int evalStm(const Position& pos, const SearchContext& ctx, int ply) {
  if (ctx.useNNUE && ctx.nnue && ply >= 0 && ply < MAX_PLY) {
    return ctx.nnue->evaluateStm(pos, PLY.nnueAcc[static_cast<std::size_t>(ply)]);
  }
  return hceEvalStm(pos);
}

static inline int historyScore(const SearchContext& ctx, const Move& m) {
  if (m.from >= SQ_N || m.to >= SQ_N) return 0;
  return ctx.history[historyIndex(m)];
}

static inline void recordQuietCutoff(SearchContext& ctx, const Move& m, int ply, int depth) {
  if (ply < 0 || ply >= MAX_PLY) return;

  // Killer moves (two slots per ply).
  if (!sameMove(ctx.killers[ply][0], m)) {
    ctx.killers[ply][1] = ctx.killers[ply][0];
    ctx.killers[ply][0] = m;
  }

  // History heuristic.
  if (m.from < SQ_N && m.to < SQ_N) {
    const std::size_t idx = historyIndex(m);
    const int bonus = depth * depth;
    int v = ctx.history[idx] + bonus;
    if (v > 1'000'000) v = 1'000'000;
    ctx.history[idx] = v;
  }
}

static inline int orderScore(const Position& pos, const Move& m, const Move& ttBest, const SearchContext& ctx, int ply) {
  if (ttBest.to != SQ_NONE && sameMove(m, ttBest)) return 1'000'000'000;

  int sc = moveHeuristic(pos, m);
  if (isQuietMove(pos, m)) {
    if (sameMove(m, ctx.killers[ply][0])) sc += 900'000;
    else if (sameMove(m, ctx.killers[ply][1])) sc += 800'000;
    sc += historyScore(ctx, m);
  }
  return sc;
}

static std::vector<Move> extractPV(Position pos, std::uint64_t key, int maxLen) {
  std::vector<Move> pv;
  if (maxLen <= 0) return pv;
  pv.reserve(static_cast<std::size_t>(maxLen));

  std::array<std::uint64_t, MAX_PLY> seen{};
  int seenN = 0;

  for (int i = 0; i < maxLen; ++i) {
    if (pos.gameOver()) break;

    // Repetition guard (very cheap): stop if we see a key twice in this PV walk.
    for (int j = 0; j < seenN; ++j) {
      if (seen[static_cast<std::size_t>(j)] == key) return pv;
    }
    if (seenN < MAX_PLY) seen[static_cast<std::size_t>(seenN++)] = key;

    const TTEntry& e = ttSlot(key);
    if (e.key != key || e.best.to == SQ_NONE) break;

    MoveList moves;
    pos.generateMoves(moves);
    bool found = false;
    Move m = nullMove();
    for (std::uint32_t k = 0; k < moves.size; ++k) {
      if (sameMove(moves.buf[k], e.best)) {
        m = moves.buf[k];
        found = true;
        break;
      }
    }
    if (!found) break;

    pv.push_back(m);
    Undo u;
    pos.makeMove(m, u);
    key = hashAfterMake(key, pos, u);
  }

  return pv;
}

static int quiescence(Position& pos, int alpha, int beta, SearchContext& ctx, int ply, std::uint64_t key, int qDepth) {
  ++ctx.nodes;
  if (ply > ctx.seldepth) ctx.seldepth = ply;
  if (ctx.shouldStop()) return 0;

  if (pos.gameOver()) return mateScore(ply); // side-to-move is the winner in our state model
  if (ply >= MAX_PLY) return evalStm(pos, ctx, ply); // safety against pathological cycles

  // Claimable threefold draw is an available action at this node.
  if (ply > 0 && pos.isRepetition()) {
    if (0 > alpha) alpha = 0;
    if (alpha >= beta) return alpha;
  }

  const int stand = evalStm(pos, ctx, ply);
  if (stand >= beta) return beta;
  if (stand > alpha) alpha = stand;
  if (qDepth <= 0) return alpha;

  MoveList& moves = PLY.moves[static_cast<std::size_t>(ply)];
  generateNoisyMoves(pos, moves);
  if (moves.empty()) return alpha;

  auto& scores = PLY.scores[static_cast<std::size_t>(ply)];
  for (std::uint32_t i = 0; i < moves.size; ++i) scores[i] = moveHeuristic(pos, moves.buf[i]);

  for (std::uint32_t i = 0; i < moves.size; ++i) {
    // Select best remaining by heuristic.
    std::uint32_t bestIdx = i;
    int bestSc = scores[i];
    for (std::uint32_t j = i + 1; j < moves.size; ++j) {
      const int s = scores[j];
      if (s > bestSc) {
        bestSc = s;
        bestIdx = j;
      }
    }
    if (bestIdx != i) {
      std::swap(moves.buf[i], moves.buf[bestIdx]);
      std::swap(scores[i], scores[bestIdx]);
    }

    const Move m = moves.buf[i];
    Undo u;
    const std::uint64_t key0 = key;
    if (ctx.useNNUE && ply + 1 < MAX_PLY) PLY.nnueAcc[static_cast<std::size_t>(ply + 1)] = PLY.nnueAcc[static_cast<std::size_t>(ply)];
    pos.makeMove(m, u);
    if (ctx.useNNUE && ply + 1 < MAX_PLY) ctx.nnue->applyDeltaAfterMove(PLY.nnueAcc[static_cast<std::size_t>(ply + 1)], pos, u);
    key = hashAfterMake(key, pos, u);

    const int score = pos.gameOver() ? mateScore(ply + 1) : -quiescence(pos, -beta, -alpha, ctx, ply + 1, key, qDepth - 1);

    pos.undoMove(u);
    key = key0;

    if (ctx.aborted) return 0;
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

static int negamax(Position& pos, int depth, int alpha, int beta, SearchContext& ctx, int ply, std::uint64_t key, bool pvNode) {
  // Threefold repetition is a *claimable* draw (not forced). Treat it as an available
  // action with score 0: the side-to-move can always claim if it's beneficial, but may
  // also choose to play on (e.g. when winning).
  const bool canClaimDraw = (ply > 0) && pos.isRepetition();
  if (depth <= 0) {
    const int q = quiescence(pos, alpha, beta, ctx, ply, key, QS_MAX_DEPTH);
    return canClaimDraw ? std::max(0, q) : q;
  }

  ++ctx.nodes;
  if (ply > ctx.seldepth) ctx.seldepth = ply;
  if (ctx.shouldStop()) return 0;

  if (pos.gameOver()) return mateScore(ply);
  if (ply >= MAX_PLY) return evalStm(pos, ctx, ply);

  const int alphaOrig = alpha;

  // If a draw claim is available, the side-to-move can secure at least 0.
  int best = -INF;
  if (canClaimDraw) {
    best = 0;
    if (best > alpha) alpha = best;
    if (alpha >= beta) return best;
  }

  // Mate-distance pruning.
  alpha = std::max(alpha, -MATE + ply);
  beta = std::min(beta, MATE - ply - 1);
  if (alpha >= beta) return alpha;

  // Transposition table probe.
  Move ttBest = nullMove();
  if (ctx.useTT) {
    const TTEntry& e = ttSlot(key);
    if (e.key == key) {
      ttBest = e.best;
      if (e.depth >= depth) {
        int ttScore = scoreFromTT(e.score, ply);
        // If a draw claim is available at this node, the value cannot be below 0.
        if (canClaimDraw && ttScore < 0) ttScore = 0;
        // When a draw claim is available, an Exact=0 stored score can be history-dependent;
        // don't let it short-circuit a potentially winning continuation.
        if (e.flag == TTFlag::Exact) {
          if (!canClaimDraw || ttScore != 0) return ttScore;
        }
        if (e.flag == TTFlag::Lower && ttScore >= beta) return ttScore;
        if (e.flag == TTFlag::Upper && ttScore <= alpha) return ttScore;
      }
    }
  }

  int staticEval = 0;
  bool haveStaticEval = false;
  auto getStaticEval = [&]() -> int {
    if (!haveStaticEval) {
      staticEval = evalStm(pos, ctx, ply);
      haveStaticEval = true;
    }
    return staticEval;
  };

  // NNUE pruning policy: keep it more conservative (especially for newly trained nets).
  const bool conservativeEvalPruning = ctx.useNNUE;

  // Razoring (very shallow): if we are far below alpha, go straight to quiescence.
  if (!pvNode && depth <= 2 && !conservativeEvalPruning) {
    const int ev = getStaticEval();
    const int razorMargin = 220 + (depth - 1) * 180;
    if (ev + razorMargin <= alpha) return quiescence(pos, alpha, beta, ctx, ply, key, QS_MAX_DEPTH);
  }

  // Reverse futility pruning (fail-high) at shallow depth.
  if (!pvNode && depth <= 2 && !conservativeEvalPruning) {
    const int ev = getStaticEval();
    const int margin = 160 + depth * 120;
    if (ev - margin >= beta) return ev;
  }

  // Null-move pruning (disabled in very low material to reduce zugzwang risk).
  if (!pvNode && depth >= (ctx.useNNUE ? 4 : 3) && ply > 0 && nonSovPieceCount(pos, pos.turn()) >= (ctx.useNNUE ? 4 : 3)) {
    const int R = ctx.useNNUE ? (1 + ((depth >= 7) ? 1 : 0)) : (2 + ((depth >= 6) ? 1 : 0));
    NullUndo nu;
    if (ctx.useNNUE && ply + 1 < MAX_PLY) PLY.nnueAcc[static_cast<std::size_t>(ply + 1)] = PLY.nnueAcc[static_cast<std::size_t>(ply)];
    pos.makeNullMove(nu);
    if (ctx.useNNUE && ply + 1 < MAX_PLY) ctx.nnue->applyDeltaAfterNullMove(PLY.nnueAcc[static_cast<std::size_t>(ply + 1)], pos, nu);
    const std::uint64_t nullKey = key ^ zobrist().turn;
    const int score = -negamax(pos, depth - 1 - R, -beta, -(beta - 1), ctx, ply + 1, nullKey, false);
    pos.undoNullMove(nu);
    if (ctx.aborted) return 0;
    if (score >= beta) return beta;
  }

  MoveList& moves = PLY.moves[static_cast<std::size_t>(ply)];
  pos.generateMoves(moves);
  if (moves.empty()) return getStaticEval();

  // Score moves once, then do lazy selection-ordering.
  auto& scores = PLY.scores[static_cast<std::size_t>(ply)];
  for (std::uint32_t i = 0; i < moves.size; ++i) scores[i] = orderScore(pos, moves.buf[i], ttBest, ctx, ply);

  Move bestMove = moves.buf[0];

  for (std::uint32_t i = 0; i < moves.size; ++i) {
    // Select best remaining by score.
    std::uint32_t bestIdx = i;
    int bestSc = scores[i];
    for (std::uint32_t j = i + 1; j < moves.size; ++j) {
      const int s = scores[j];
      if (s > bestSc) {
        bestSc = s;
        bestIdx = j;
      }
    }
    if (bestIdx != i) {
      std::swap(moves.buf[i], moves.buf[bestIdx]);
      std::swap(scores[i], scores[bestIdx]);
    }

    const Move m = moves.buf[i];
    const bool quiet = isQuietMove(pos, m);

    // Futility: at depth 1, skip late quiet moves if we cannot raise alpha.
    if (!pvNode && depth == 1 && quiet) {
      const int ev = getStaticEval();
      const int margin = ctx.useNNUE ? 340 : 220;
      if (ev + margin <= alpha) continue;
    }

    // Late-move pruning (very shallow): after enough quiet moves, skip more quiet moves
    // when we're not improving alpha. This helps speed in locked wall endgames.
    if (!pvNode && depth == 2 && quiet) {
      const int ev = getStaticEval();
      const std::uint32_t moveCount = ctx.useNNUE ? 32u : 20u;
      const int margin = ctx.useNNUE ? 200 : 140;
      if (i >= moveCount && ev + margin <= alpha) continue;
    }

    Undo u;
    const std::uint64_t key0 = key;
    if (ctx.useNNUE && ply + 1 < MAX_PLY) PLY.nnueAcc[static_cast<std::size_t>(ply + 1)] = PLY.nnueAcc[static_cast<std::size_t>(ply)];
    pos.makeMove(m, u);
    if (ctx.useNNUE && ply + 1 < MAX_PLY) ctx.nnue->applyDeltaAfterMove(PLY.nnueAcc[static_cast<std::size_t>(ply + 1)], pos, u);
    key = hashAfterMake(key, pos, u);

    int score = 0;
    if (pos.gameOver()) {
      score = mateScore(ply + 1);
    } else {
      const int newDepth = depth - 1;

      if (pvNode && i == 0) {
        // PV move: full window.
        score = -negamax(pos, newDepth, -beta, -alpha, ctx, ply + 1, key, true);
      } else {
        // Non-PV: PVS null window, with LMR for late quiet moves.
        int searchDepth = newDepth;
        const bool doLMR = (!pvNode && quiet && depth >= 3 && i >= 4);
        if (doLMR) {
          const int r = 1 + ((i >= 8) ? 1 : 0) + ((depth >= 6) ? 1 : 0);
          searchDepth = newDepth - r;
          if (searchDepth < 1) searchDepth = 1;
        }

        score = -negamax(pos, searchDepth, -(alpha + 1), -alpha, ctx, ply + 1, key, false);
        if (!ctx.aborted) {
          // If reduced search (or null-window) indicates improvement, re-search deeper / wider.
          if (score > alpha) {
            if (doLMR && searchDepth != newDepth) {
              score = -negamax(pos, newDepth, -(alpha + 1), -alpha, ctx, ply + 1, key, false);
            }
            if (score > alpha && score < beta) {
              score = -negamax(pos, newDepth, -beta, -alpha, ctx, ply + 1, key, true);
            }
          }
        }
      }
    }

    pos.undoMove(u);
    key = key0;

    if (ctx.aborted) return 0;

    if (score > best) {
      best = score;
      bestMove = m;
    }
    if (best > alpha) alpha = best;

    if (alpha >= beta) {
      if (quiet) recordQuietCutoff(ctx, m, ply, depth);
      break;
    }
  }

  // Store to TT.
  if (ctx.useTT) {
    TTEntry& out = ttSlot(key);
    const TTFlag flag = (best <= alphaOrig) ? TTFlag::Upper : (best >= beta) ? TTFlag::Lower : TTFlag::Exact;
    if (out.key == 0 || out.key == key || depth >= out.depth) {
      out.key = key;
      out.depth = depth;
      out.flag = flag;
      out.score = scoreToTT(best, ply);
      out.best = bestMove;
    }
  }

  return best;
}

struct RootOut {
  int score = -INF;
  Move best = nullMove();
};

static RootOut searchRoot(Position& pos, std::uint64_t rootKey, MoveList& moves, int depth, int alpha, int beta, SearchContext& ctx) {
  RootOut out;
  if (moves.empty()) return out;

  // Root move ordering: heuristic + TT + killers/history.
  Move ttBest = nullMove();
  if (ctx.useTT) {
    const TTEntry& e = ttSlot(rootKey);
    if (e.key == rootKey) ttBest = e.best;
  }

  std::array<int, 4096> scores{};
  for (std::uint32_t i = 0; i < moves.size; ++i) scores[i] = orderScore(pos, moves.buf[i], ttBest, ctx, 0);

  int bestScore = -INF;
  Move bestMove = moves.buf[0];
  int alpha0 = alpha;

  for (std::uint32_t i = 0; i < moves.size; ++i) {
    // Select next best move by ordering score.
    std::uint32_t bestIdx = i;
    int bestSc = scores[i];
    for (std::uint32_t j = i + 1; j < moves.size; ++j) {
      const int s = scores[j];
      if (s > bestSc) {
        bestSc = s;
        bestIdx = j;
      }
    }
    if (bestIdx != i) {
      std::swap(moves.buf[i], moves.buf[bestIdx]);
      std::swap(scores[i], scores[bestIdx]);
    }

    const Move m = moves.buf[i];
    Undo u;
    if (ctx.useNNUE) PLY.nnueAcc[1] = PLY.nnueAcc[0];
    pos.makeMove(m, u);
    if (ctx.useNNUE) ctx.nnue->applyDeltaAfterMove(PLY.nnueAcc[1], pos, u);
    const std::uint64_t childKey = hashAfterMake(rootKey, pos, u);

    int score = 0;
    if (pos.gameOver()) {
      score = mateScore(1);
    } else if (i == 0) {
      score = -negamax(pos, depth - 1, -beta, -alpha, ctx, 1, childKey, true);
    } else {
      score = -negamax(pos, depth - 1, -(alpha + 1), -alpha, ctx, 1, childKey, false);
      if (!ctx.aborted && score > alpha && score < beta) {
        score = -negamax(pos, depth - 1, -beta, -alpha, ctx, 1, childKey, true);
      }
    }

    pos.undoMove(u);
    if (ctx.aborted) break;

    if (score > bestScore) {
      bestScore = score;
      bestMove = m;
    }
    if (score > alpha) alpha = score;
    if (alpha >= beta) {
      // Aspiration fail-high can trigger a root cutoff.
      if (isQuietMove(pos, m)) recordQuietCutoff(ctx, m, 0, depth);
      break;
    }
  }

  // Store root for ordering/PV reconstruction.
  if (ctx.useTT) {
    TTEntry& r = ttSlot(rootKey);
    const TTFlag flag = (bestScore <= alpha0) ? TTFlag::Upper : (bestScore >= beta) ? TTFlag::Lower : TTFlag::Exact;
    r.key = rootKey;
    r.depth = depth;
    r.flag = flag;
    r.score = scoreToTT(bestScore, 0);
    r.best = bestMove;
  }

  out.score = bestScore;
  out.best = bestMove;
  return out;
}

SearchResult searchBestMove(Position& pos, const SearchOptions& opt) {
  if (opt.useTT) ensureTT();

  SearchResult res;
  SearchContext ctx;
  ctx.limits = opt.limits;
  ctx.onInfo = opt.onInfo;
  ctx.stop = opt.stop;
  ctx.evalBackend = opt.evalBackend;
  ctx.nnue = opt.nnue;
  ctx.useNNUE = (opt.evalBackend == EvalBackend::NNUE) && (opt.nnue != nullptr) && opt.nnue->loaded();
  ctx.useTT = opt.useTT;
  ctx.nodeLimit = opt.limits.nodeLimit;
  ctx.useTime = opt.limits.timeLimitMs != 0;

  const auto t0 = std::chrono::steady_clock::now();
  ctx.start = t0;
  if (ctx.useTime) ctx.end = t0 + std::chrono::milliseconds(opt.limits.timeLimitMs);
  ctx.resetHeuristics();

  int maxDepth = opt.limits.depth;
  if (maxDepth <= 0) maxDepth = 1;
  if (maxDepth > MAX_PLY - 1) maxDepth = MAX_PLY - 1;

  MoveList rootMoves;
  pos.generateMoves(rootMoves);
  if (rootMoves.empty()) {
    res.best = nullMove();
    res.score = 0;
    res.nodes = 0;
    res.seconds = 0.0;
    return res;
  }

  if (ctx.useNNUE) ctx.nnue->initAccumulator(pos, PLY.nnueAcc[0]);

  const std::uint64_t rootKey = hashPosition(pos);

  Move bestMove = rootMoves.buf[0];
  int bestScore = -INF;

  int prevScore = 0;
  int lastCompletedDepth = 0;

  for (int curDepth = 1; curDepth <= maxDepth; ++curDepth) {
    if (ctx.shouldStop()) break;
    ctx.seldepth = 0;

    int alpha = -INF;
    int beta = INF;

    // Aspiration windows after depth 1.
    int window = (curDepth <= 2) ? 140 : 90;
    if (curDepth > 1) {
      alpha = prevScore - window;
      beta = prevScore + window;
    }

    RootOut iter;
    while (true) {
      iter = searchRoot(pos, rootKey, rootMoves, curDepth, alpha, beta, ctx);
      if (ctx.aborted) break;

      if (curDepth == 1) break;
      if (iter.score <= alpha) {
        // fail-low: widen downward
        alpha = -INF;
        window *= 2;
        beta = iter.score + window;
        continue;
      }
      if (iter.score >= beta) {
        // fail-high: widen upward
        beta = INF;
        window *= 2;
        alpha = iter.score - window;
        continue;
      }
      break;
    }

    if (ctx.aborted) break;

    bestMove = iter.best;
    bestScore = iter.score;
    prevScore = bestScore;
    lastCompletedDepth = curDepth;

    if (ctx.onInfo) {
      SearchInfo info;
      info.depth = curDepth;
      info.seldepth = ctx.seldepth;
      info.score = bestScore;
      info.nodes = ctx.nodes;
      info.timeMs = ctx.elapsedMs();
      info.best = bestMove;
      if (ctx.useTT) info.pv = extractPV(pos, rootKey, std::min(MAX_PLY - 1, curDepth + 16));
      ctx.onInfo(info);
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const std::chrono::duration<double> dt = t1 - t0;

  // If we never finished a depth (very short time limit), fall back to TT (if available).
  if (lastCompletedDepth == 0) {
    if (ctx.useTT) {
      const TTEntry& e = ttSlot(rootKey);
      if (e.key == rootKey) {
        if (e.best.to != SQ_NONE) {
          // Validate TT move at root (defensive against collisions).
          bool ok = false;
          for (std::uint32_t i = 0; i < rootMoves.size; ++i) {
            if (sameMove(rootMoves.buf[i], e.best)) {
              ok = true;
              break;
            }
          }
          if (ok) bestMove = e.best;
        }
        bestScore = scoreFromTT(e.score, 0);
      } else {
        bestMove = rootMoves.buf[0];
        bestScore = evalStm(pos, ctx, 0);
      }
    } else {
      bestMove = rootMoves.buf[0];
      bestScore = evalStm(pos, ctx, 0);
    }
  }

  res.best = bestMove;
  res.score = bestScore;
  res.nodes = ctx.nodes;
  res.seconds = dt.count();
  return res;
}

SearchResult searchBestMove(Position& pos, int depth) {
  SearchOptions opt;
  opt.limits.depth = depth;
  return searchBestMove(pos, opt);
}

int evaluatePositionStm(const Position& pos, EvalBackend backend, const NNUE* nnue) {
  if (backend == EvalBackend::NNUE && nnue && nnue->loaded()) {
    NNUE::Accumulator acc;
    nnue->initAccumulator(pos, acc);
    return nnue->evaluateStm(pos, acc);
  }
  return hceEvalStm(pos);
}

} // namespace citadel

