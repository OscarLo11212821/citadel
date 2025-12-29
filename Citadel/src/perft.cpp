#include "citadel/perft.hpp"

#include <chrono>

namespace citadel {

std::uint64_t perft(Position& pos, int depth) {
  if (depth <= 0) return 1;

  MoveList moves;
  pos.generateMoves(moves);
  if (moves.empty()) return 0;
  if (depth == 1) return moves.size;

  std::uint64_t nodes = 0;
  for (std::uint32_t i = 0; i < moves.size; ++i) {
    Undo u;
    pos.makeMove(moves.buf[i], u);
    nodes += perft(pos, depth - 1);
    pos.undoMove(u);
  }
  return nodes;
}

std::vector<std::pair<Move, std::uint64_t>> perftDivide(Position& pos, int depth) {
  std::vector<std::pair<Move, std::uint64_t>> out;
  if (depth <= 0) return out;

  MoveList moves;
  pos.generateMoves(moves);
  out.reserve(moves.size);

  for (std::uint32_t i = 0; i < moves.size; ++i) {
    Undo u;
    pos.makeMove(moves.buf[i], u);
    const std::uint64_t n = perft(pos, depth - 1);
    pos.undoMove(u);
    out.push_back({moves.buf[i], n});
  }
  return out;
}

PerftStats perftTimed(Position& pos, int depth) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::uint64_t nodes = perft(pos, depth);
  const auto t1 = std::chrono::steady_clock::now();

  const std::chrono::duration<double> dt = t1 - t0;
  PerftStats st;
  st.nodes = nodes;
  st.seconds = dt.count();
  st.nps = (st.seconds > 0.0) ? (static_cast<double>(nodes) / st.seconds) : 0.0;
  return st;
}

} // namespace citadel

