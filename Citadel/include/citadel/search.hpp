#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "citadel/move.hpp"
#include "citadel/position.hpp"

namespace citadel {

class NNUE;

struct SearchLimits {
  int depth = 4;                 // max depth in plies (>=1)
  std::uint64_t nodeLimit = 0;   // 0 = unlimited
  std::uint64_t timeLimitMs = 0; // 0 = unlimited
};

struct SearchInfo {
  int depth = 0;
  int seldepth = 0;
  int score = 0; // centipawn-like, from side-to-move perspective
  std::uint64_t nodes = 0;
  std::uint64_t timeMs = 0;
  Move best = nullMove();
  std::vector<Move> pv;
};

using SearchInfoCallback = std::function<void(const SearchInfo&)>;

enum class EvalBackend : std::uint8_t { HCE = 0, NNUE = 1 };

struct SearchOptions {
  SearchLimits limits{};
  SearchInfoCallback onInfo{};
  std::atomic_bool* stop = nullptr; // optional external stop signal

  // Evaluation selection.
  EvalBackend evalBackend = EvalBackend::HCE;
  const NNUE* nnue = nullptr; // required when evalBackend == NNUE

  // Transposition table (TT) usage. Set false when calling search concurrently from multiple
  // threads (Citadel's TT is single-threaded today).
  bool useTT = true;
};

struct SearchResult {
  Move best = nullMove();
  int score = 0; // centipawn-like, from side-to-move perspective
  std::uint64_t nodes = 0;
  double seconds = 0.0;
};

// Transposition table controls (useful for UCI).
void clearTranspositionTable();
void setTranspositionTableSizeMB(std::size_t mb);
[[nodiscard]] std::size_t transpositionTableSizeMB();

[[nodiscard]] SearchResult searchBestMove(Position& pos, const SearchOptions& opt);
[[nodiscard]] SearchResult searchBestMove(Position& pos, int depth);

// Evaluate a position without searching.
// Returns a centipawn-like score from side-to-move perspective.
[[nodiscard]] int evaluatePositionStm(const Position& pos, EvalBackend backend = EvalBackend::HCE, const NNUE* nnue = nullptr);

} // namespace citadel

