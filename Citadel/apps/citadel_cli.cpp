#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cctype>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "citadel/perft.hpp"
#include "citadel/nnue.hpp"
#include "citadel/search.hpp"

using citadel::Move;
using citadel::MoveList;
using citadel::Position;

static constexpr const char* DEFAULT_NNUE_FILE = "/Users/parent/Desktop/Citadel/nnue_2912251254.cnue";

static std::string toLowerCopy(std::string_view sv);
static std::optional<std::string> argValue(int argc, char** argv, std::string_view key);

static std::optional<citadel::EvalBackend> parseEvalBackend(std::string_view s) {
  const std::string v = toLowerCopy(s);
  if (v == "hce") return citadel::EvalBackend::HCE;
  if (v == "nnue") return citadel::EvalBackend::NNUE;
  return std::nullopt;
}

struct EvalContext {
  citadel::EvalBackend backend = citadel::EvalBackend::NNUE;
  citadel::NNUE nnue{};
  std::string nnueFile{};

  [[nodiscard]] const citadel::NNUE* nnuePtr() const {
    if (backend == citadel::EvalBackend::NNUE && nnue.loaded()) return &nnue;
    return nullptr;
  }
};

static EvalContext loadEvalForCommand(int argc, char** argv) {
  EvalContext ec;

  // Default behavior: NNUE with a built-in default model path, fallback to HCE if load fails.
  ec.backend = citadel::EvalBackend::NNUE;
  ec.nnueFile = DEFAULT_NNUE_FILE;

  if (auto v = argValue(argc, argv, "--eval")) {
    if (auto eb = parseEvalBackend(*v)) ec.backend = *eb;
    else std::cerr << "warning: unknown --eval '" << *v << "', using default\n";
  }
  if (auto v = argValue(argc, argv, "--nnuefile")) {
    if (!v->empty()) ec.nnueFile = *v;
  }

  if (ec.backend == citadel::EvalBackend::NNUE) {
    if (!ec.nnue.loadFromFile(ec.nnueFile)) {
      std::cerr << "warning: nnue load failed: " << ec.nnue.lastError() << " (falling back to HCE)\n";
      ec.backend = citadel::EvalBackend::HCE;
    }
  }

  return ec;
}

static void usage(std::string_view exe) {
  std::cerr << "Usage:\n"
            << "  " << exe << " uci\n"
            << "  " << exe << " perft <depth> [--fen <fen>] [--divide]\n"
            << "  " << exe << " bestmove [--depth N] [--fen <fen>] [--eval hce|nnue] [--nnuefile <path>]\n"
            << "  " << exe << " play [--engine white|black|none] [--depth N] [--fen <fen>] [--pgn <file>] [--append] [--eval hce|nnue] [--nnuefile <path>]\n"
            << "  " << exe << " selfplay [--depth N] [--maxplies N] [--fen <fen>] [--pgn <file>] [--append] [--eval hce|nnue] [--nnuefile <path>]\n"
            << "  " << exe << " datagen --out <file> [--samples N] [--depth N] [--maxplies N] [--fen <fen>] [--append] [--seed N]\n"
            << "                 [--random-move-prob P] [--randomize-start N] [--threads N] [--fenfile <file>] [--eval hce|nnue] [--nnuefile <path>]\n"
            << "  " << exe << " review [--depth N] [--pgn <file>|-] [--eval hce|nnue] [--nnuefile <path>]\n"
            << "       (omit --pgn or use '-' to read PGN from stdin)\n";
}

static std::optional<std::string> argValue(int argc, char** argv, std::string_view key) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string_view(argv[i]) == key) return std::string(argv[i + 1]);
  }
  return std::nullopt;
}

static bool hasFlag(int argc, char** argv, std::string_view key) {
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == key) return true;
  }
  return false;
}

static int intArg(int argc, char** argv, std::string_view key, int def) {
  if (auto v = argValue(argc, argv, key)) return std::atoi(v->c_str());
  return def;
}

static double doubleArg(int argc, char** argv, std::string_view key, double def) {
  if (auto v = argValue(argc, argv, key)) return std::atof(v->c_str());
  return def;
}

static Position loadPositionFromArgs(int argc, char** argv) {
  if (auto fen = argValue(argc, argv, "--fen")) return Position::fromFEN(*fen);
  return Position::initial();
}

static std::string pgnEscape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (const char ch : s) {
    if (ch == '\\' || ch == '"') out.push_back('\\');
    out.push_back(ch);
  }
  return out;
}

static std::string todayPgnDate() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  tm = *std::localtime(&t);
#endif
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(4) << (tm.tm_year + 1900) << '.' << std::setw(2) << (tm.tm_mon + 1) << '.' << std::setw(2) << tm.tm_mday;
  return oss.str();
}

static int parseFenFullmove(std::string_view fen) {
  const auto sp = fen.find_last_of(' ');
  if (sp == std::string_view::npos) return 1;
  const std::string_view n = fen.substr(sp + 1);
  const int v = std::atoi(std::string(n).c_str());
  return (v > 0) ? v : 1;
}

static std::string resultTokenFromWinner(const std::optional<citadel::Color>& w, bool draw) {
  if (draw) return "1/2-1/2";
  if (!w) return "*";
  return (*w == citadel::Color::White) ? "1-0" : "0-1";
}

static std::string terminationString(const Position& pos, bool draw, bool abandoned) {
  if (abandoned) return "Abandoned";
  if (draw) return "MoveLimit";
  if (!pos.gameOver()) return "Unterminated";
  return (pos.winReason() == citadel::WinReason::Regicide) ? "Regicide" : "Entombment";
}

static std::string formatMovetext(const std::vector<Move>& moves, const Position& startPos, int startFullmove,
                                  std::string_view resultTok) {
  std::ostringstream oss;
  int moveNo = startFullmove;
  citadel::Color stm = startPos.turn();
  std::size_t lineLen = 0;

  auto emit = [&](std::string_view tok) {
    if (tok.empty()) return;
    const std::size_t add = tok.size() + ((lineLen == 0) ? 0 : 1);
    if (lineLen != 0 && lineLen + add > 80) {
      oss << "\n";
      lineLen = 0;
    }
    if (lineLen != 0) {
      oss << ' ';
      ++lineLen;
    }
    oss << tok;
    lineLen += tok.size();
  };

  // If Black to move initially, PGN uses "N... <blackmove>".
  if (stm == citadel::Color::Black) {
    emit(std::to_string(moveNo) + "...");
  }

  for (std::size_t ply = 0; ply < moves.size(); ++ply) {
    if (stm == citadel::Color::White) emit(std::to_string(moveNo) + ".");
    const std::string tok = citadel::moveToPgnToken(moves[ply]);
    emit(tok);

    stm = citadel::other(stm);
    if (stm == citadel::Color::White) ++moveNo;
  }

  emit(std::string(resultTok));
  oss << "\n";
  return oss.str();
}

static void writePgnGame(std::ostream& os, std::string_view eventName, std::string_view whiteName, std::string_view blackName, std::string_view site,
                         std::string_view date, std::string_view round, std::string_view startFen, std::string_view resultTok,
                         std::string_view termination, const std::vector<Move>& moves, const Position& startPos) {
  os << "[Event \"" << pgnEscape(eventName) << "\"]\n";
  os << "[Site \"" << pgnEscape(site) << "\"]\n";
  os << "[Date \"" << pgnEscape(date) << "\"]\n";
  os << "[Round \"" << pgnEscape(round) << "\"]\n";
  os << "[White \"" << pgnEscape(whiteName) << "\"]\n";
  os << "[Black \"" << pgnEscape(blackName) << "\"]\n";
  os << "[Result \"" << pgnEscape(resultTok) << "\"]\n";
  os << "[Variant \"Citadel\"]\n";
  os << "[Termination \"" << pgnEscape(termination) << "\"]\n";
  os << "[SetUp \"1\"]\n";
  os << "[FEN \"" << pgnEscape(startFen) << "\"]\n";
  os << "[PlyCount \"" << moves.size() << "\"]\n";
  os << "\n";

  const int startFullmove = parseFenFullmove(startFen);
  os << formatMovetext(moves, startPos, startFullmove, resultTok);
  os << "\n";
}

static void maybeWritePgnToFile(const std::optional<std::string>& pgnPath, bool append, std::string_view eventName, std::string_view whiteName,
                               std::string_view blackName, std::string_view startFen, std::string_view resultTok, std::string_view termination,
                               const std::vector<Move>& moves, const Position& startPos) {
  if (!pgnPath) return;

  std::ofstream f;
  f.open(*pgnPath, append ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc));
  if (!f) throw std::runtime_error("Failed to open PGN file for writing: " + *pgnPath);

  writePgnGame(f, eventName, whiteName, blackName, "local", todayPgnDate(), "1", startFen, resultTok, termination, moves, startPos);
}

static void cmdPerft(int argc, char** argv) {
  if (argc < 3) throw std::runtime_error("perft: missing depth");
  const int depth = std::atoi(argv[2]);
  Position pos = loadPositionFromArgs(argc, argv);

  std::cout << pos.pretty() << "\n";
  std::cout << "FEN: " << pos.toFEN() << "\n\n";

  if (hasFlag(argc, argv, "--divide")) {
    const auto rows = citadel::perftDivide(pos, depth);
    std::uint64_t total = 0;
    for (const auto& [m, n] : rows) {
      std::cout << citadel::moveToString(m) << "  " << n << "\n";
      total += n;
    }
    std::cout << "Total: " << total << "\n";
  } else {
    const auto st = citadel::perftTimed(pos, depth);
    std::cout << "Nodes: " << st.nodes << "\n";
    std::cout << "Time : " << st.seconds << " s\n";
    std::cout << "NPS  : " << static_cast<std::uint64_t>(st.nps) << "\n";
  }
}

static void cmdBestmove(int argc, char** argv) {
  const int depth = intArg(argc, argv, "--depth", 4);
  Position pos = loadPositionFromArgs(argc, argv);

  std::cout << pos.pretty() << "\n";
  std::cout << "FEN: " << pos.toFEN() << "\n\n";

  EvalContext ec = loadEvalForCommand(argc, argv);
  citadel::SearchOptions opt;
  opt.limits.depth = depth;
  opt.evalBackend = ec.backend;
  opt.nnue = ec.nnuePtr();
  auto r = citadel::searchBestMove(pos, opt);
  std::cout << "bestmove " << citadel::moveToString(r.best) << "\n";
  std::cout << "score    " << r.score << "\n";
  std::cout << "nodes    " << r.nodes << "\n";
  std::cout << "time     " << r.seconds << " s\n";
  if (r.seconds > 0.0) std::cout << "nps      " << static_cast<std::uint64_t>(static_cast<double>(r.nodes) / r.seconds) << "\n";
}

static std::optional<citadel::Color> parseColor(std::string_view s) {
  if (s == "white") return citadel::Color::White;
  if (s == "black") return citadel::Color::Black;
  if (s == "none") return std::nullopt;
  return std::nullopt;
}

static void cmdPlay(int argc, char** argv) {
  const int depth = intArg(argc, argv, "--depth", 3);
  const std::string engineStr = argValue(argc, argv, "--engine").value_or("black");
  const auto engineSide = parseColor(engineStr);

  Position pos = loadPositionFromArgs(argc, argv);
  const std::string startFen = pos.toFEN();
  const bool append = hasFlag(argc, argv, "--append");
  const auto pgnPath = argValue(argc, argv, "--pgn");
  std::vector<Move> history;
  history.reserve(512);
  bool abandoned = false;

  EvalContext ec = loadEvalForCommand(argc, argv);
  citadel::SearchOptions opt;
  opt.limits.depth = depth;
  opt.evalBackend = ec.backend;
  opt.nnue = ec.nnuePtr();

  while (!pos.gameOver()) {
    std::cout << "\n" << pos.pretty() << "\n";
    std::cout << "FEN: " << pos.toFEN() << "\n";

    MoveList moves;
    pos.generateMoves(moves);
    if (moves.empty()) {
      std::cout << "No legal moves.\n";
      break;
    }

    if (engineSide && *engineSide == pos.turn()) {
      auto r = citadel::searchBestMove(pos, opt);
      std::cout << "Engine plays: " << citadel::moveToString(r.best) << " (score " << r.score << ")\n";
      citadel::Undo u;
      pos.makeMove(r.best, u);
      history.push_back(r.best);
      continue;
    }

    std::cout << "Moves:\n";
    for (std::uint32_t i = 0; i < moves.size; ++i) {
      std::cout << "  [" << i << "] " << citadel::moveToString(moves.buf[i]) << "\n";
    }

    std::cout << "Select move index (or 'q' to quit): ";
    std::string line;
    if (!std::getline(std::cin, line)) {
      abandoned = true;
      break;
    }
    if (line == "q" || line == "quit" || line == "exit") {
      abandoned = true;
      break;
    }

    const int idx = std::atoi(line.c_str());
    if (idx < 0 || static_cast<std::uint32_t>(idx) >= moves.size) {
      std::cout << "Invalid index.\n";
      continue;
    }

    citadel::Undo u;
    const Move chosen = moves.buf[static_cast<std::uint32_t>(idx)];
    pos.makeMove(chosen, u);
    history.push_back(chosen);
  }

  std::cout << "\n" << pos.pretty() << "\n";
  if (pos.gameOver()) {
    const auto w = pos.winner();
    if (w) {
      std::cout << "Game over. Winner: " << citadel::colorName(*w) << " (" << ((pos.winReason() == citadel::WinReason::Regicide) ? "Regicide" : "Entombment")
                << ")\n";
    }
  }

  // PGN export (optional)
  const bool draw = false;
  const std::string resultTok = resultTokenFromWinner(pos.winner(), draw);
  const std::string term = terminationString(pos, draw, abandoned);
  const std::string whiteName = engineSide && *engineSide == citadel::Color::White ? "Citadel" : "Human";
  const std::string blackName = engineSide && *engineSide == citadel::Color::Black ? "Citadel" : "Human";
  const Position startPos = Position::fromFEN(startFen);
  maybeWritePgnToFile(pgnPath, append, "Citadel Play", whiteName, blackName, startFen, resultTok, term, history, startPos);
}

static void cmdSelfplay(int argc, char** argv) {
  const int depth = intArg(argc, argv, "--depth", 3);
  const int maxPlies = intArg(argc, argv, "--maxplies", 200);
  const bool append = hasFlag(argc, argv, "--append");
  const auto pgnPath = argValue(argc, argv, "--pgn");

  Position pos = loadPositionFromArgs(argc, argv);
  const std::string startFen = pos.toFEN();
  const Position startPos = Position::fromFEN(startFen);

  EvalContext ec = loadEvalForCommand(argc, argv);
  citadel::SearchOptions opt;
  opt.limits.depth = depth;
  opt.evalBackend = ec.backend;
  opt.nnue = ec.nnuePtr();

  std::vector<Move> history;
  history.reserve(static_cast<std::size_t>((maxPlies > 0) ? maxPlies : 256));

  bool moveLimit = false;
  bool noMoves = false;
  for (int ply = 0; ply < maxPlies && !pos.gameOver(); ++ply) {
    auto r = citadel::searchBestMove(pos, opt);
    if (r.best.from == citadel::SQ_NONE) {
      noMoves = true;
      break;
    }
    std::cout << (ply + 1) << ". " << citadel::colorName(pos.turn()) << "  " << citadel::moveToString(r.best) << "  (score " << r.score << ")\n";
    citadel::Undo u;
    pos.makeMove(r.best, u);
    history.push_back(r.best);
  }
  if (!pos.gameOver() && !noMoves) moveLimit = true;

  const bool draw = moveLimit || noMoves;
  const std::string resultTok = resultTokenFromWinner(pos.winner(), draw);
  const std::string term = moveLimit ? "MoveLimit" : noMoves ? "NoMoves" : terminationString(pos, false, false);

  std::cout << "\nResult: " << resultTok << " (" << term << ")\n";

  maybeWritePgnToFile(pgnPath, append, "Citadel Self-Play", "Citadel", "Citadel", startFen, resultTok, term, history, startPos);
}

static void cmdDatagen(int argc, char** argv) {
  const int depth = intArg(argc, argv, "--depth", 3);
  const int maxPlies = intArg(argc, argv, "--maxplies", 200);
  const int samples = intArg(argc, argv, "--samples", 10'000);
  const int randomizeStart = intArg(argc, argv, "--randomize-start", 6);
  int threads = intArg(argc, argv, "--threads", 1);
  double randomMoveProb = doubleArg(argc, argv, "--random-move-prob", 0.05);
  if (randomMoveProb < 0.0) randomMoveProb = 0.0;
  if (randomMoveProb > 1.0) randomMoveProb = 1.0;

  const bool append = hasFlag(argc, argv, "--append");
  const auto outPath = argValue(argc, argv, "--out");
  if (!outPath) throw std::runtime_error("datagen: missing required --out <file>");

  const auto fenFilePath = argValue(argc, argv, "--fenfile");

  int seed = intArg(argc, argv, "--seed", 0);
  if (seed == 0) seed = static_cast<int>(std::time(nullptr));

  if (samples <= 0) throw std::runtime_error("datagen: --samples must be > 0");
  if (depth <= 0) throw std::runtime_error("datagen: --depth must be > 0");
  if (maxPlies <= 0) throw std::runtime_error("datagen: --maxplies must be > 0");

  const Position base = loadPositionFromArgs(argc, argv);
  const std::string baseFen = base.toFEN();

  EvalContext ec = loadEvalForCommand(argc, argv);

  if (threads <= 0) {
    threads = static_cast<int>(std::thread::hardware_concurrency());
    if (threads <= 0) threads = 1;
  }

  auto trimCopy = [](std::string_view sv) -> std::string {
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) sv.remove_prefix(1);
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) sv.remove_suffix(1);
    return std::string(sv);
  };

  std::vector<std::string> startFens;
  if (fenFilePath) {
    std::ifstream f(*fenFilePath);
    if (!f) throw std::runtime_error("datagen: failed to open fenfile: " + *fenFilePath);
    std::string line;
    while (std::getline(f, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      const std::size_t hash = line.find('#');
      if (hash != std::string::npos) line.resize(hash);
      const std::string fen = trimCopy(line);
      if (fen.empty()) continue;
      startFens.push_back(fen);
    }
    if (startFens.empty()) throw std::runtime_error("datagen: fenfile contains no FENs: " + *fenFilePath);
  }

  std::ofstream out;
  out.open(*outPath, append ? std::ios::out | std::ios::app : std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("datagen: failed to open output file");

  if (!append) {
    out << "# Citadel NNUE training data\n";
    out << "# Format: <FEN> | <stm> <eval>\n";
    out << "# eval is centipawn-like from side-to-move (stm) perspective.\n";
    out << "# depth=" << depth << " maxplies=" << maxPlies << " samples=" << samples << " seed=" << seed << " randomMoveProb=" << randomMoveProb
        << " randomizeStart=" << randomizeStart << " threads=" << threads << "\n";
    out << "# base_fen=" << baseFen << "\n";
    if (fenFilePath) out << "# fenfile=" << *fenFilePath << " count=" << startFens.size() << "\n";
    out << "# eval_backend=" << ((ec.backend == citadel::EvalBackend::NNUE && ec.nnuePtr()) ? "NNUE" : "HCE") << "\n";
    if (ec.backend == citadel::EvalBackend::NNUE) out << "# nnue_file=" << ec.nnueFile << "\n";
  }

  const bool useTT = (threads == 1); // TT is single-threaded today; disable for parallel datagen.

  std::atomic<int> nextSample{0};
  std::atomic<int> gameCount{0};
  std::atomic<int> invalidStart{0};
  std::atomic<int> lastReport{0};

  std::mutex outMu;
  std::mutex errMu;

  auto reportProgress = [&]() {
    const int w = std::min(nextSample.load(std::memory_order_relaxed), samples);
    const int milestone = (w / 5000) * 5000;
    int prev = lastReport.load(std::memory_order_relaxed);
    if (milestone > prev && lastReport.compare_exchange_strong(prev, milestone)) {
      std::lock_guard<std::mutex> lk(errMu);
      std::cerr << "datagen: wrote " << milestone << " / " << samples << " samples\n";
    }
  };

  auto worker = [&](int tid) {
    const std::uint32_t s = static_cast<std::uint32_t>(seed) ^ (0x9E3779B9u * static_cast<std::uint32_t>(tid + 1));
    std::mt19937 rng(s);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    citadel::SearchOptions sopt;
    sopt.limits.depth = depth;
    sopt.evalBackend = ec.backend;
    sopt.nnue = ec.nnuePtr();
    sopt.useTT = useTT;

    std::string buffer;
    buffer.reserve(1 << 20);

    int localGames = 0;
    while (nextSample.load(std::memory_order_relaxed) < samples) {
      ++localGames;

      Position pos = base;
      if (!startFens.empty()) {
        const std::size_t idx = static_cast<std::size_t>(rng() % startFens.size());
        try {
          pos = Position::fromFEN(startFens[idx]);
        } catch (const std::exception&) {
          invalidStart.fetch_add(1, std::memory_order_relaxed);
          continue;
        }
      }

      // Randomize the start a bit to diversify openings.
      for (int i = 0; i < randomizeStart && !pos.gameOver(); ++i) {
        MoveList moves;
        pos.generateMoves(moves);
        if (moves.empty()) break;
        const std::uint32_t mi = static_cast<std::uint32_t>(rng() % moves.size);
        citadel::Undo u;
        pos.makeMove(moves.buf[mi], u);
      }

      for (int ply = 0; ply < maxPlies && !pos.gameOver(); ++ply) {
        if (nextSample.load(std::memory_order_relaxed) >= samples) break;

        const auto r = citadel::searchBestMove(pos, sopt);
        if (r.best.from == citadel::SQ_NONE) break;

        const int ticket = nextSample.fetch_add(1, std::memory_order_relaxed);
        if (ticket >= samples) break;

        const std::string fen = pos.toFEN();
        const char stm = (pos.turn() == citadel::Color::White) ? 'w' : 'b';
        buffer += fen;
        buffer += " | ";
        buffer.push_back(stm);
        buffer.push_back(' ');
        buffer += std::to_string(r.score);
        buffer.push_back('\n');

        // Choose next move: mostly bestmove, occasionally random.
        Move chosen = r.best;
        if (randomMoveProb > 0.0 && uni01(rng) < randomMoveProb) {
          MoveList moves;
          pos.generateMoves(moves);
          if (!moves.empty()) chosen = moves.buf[static_cast<std::uint32_t>(rng() % moves.size)];
        }

        citadel::Undo u;
        pos.makeMove(chosen, u);

        if (buffer.size() >= (1 << 20)) {
          std::lock_guard<std::mutex> lk(outMu);
          out << buffer;
          buffer.clear();
        }

        reportProgress();
      }
    }

    if (!buffer.empty()) {
      std::lock_guard<std::mutex> lk(outMu);
      out << buffer;
    }

    gameCount.fetch_add(localGames, std::memory_order_relaxed);
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();

  const int wrote = std::min(nextSample.load(std::memory_order_relaxed), samples);
  const int games = gameCount.load(std::memory_order_relaxed);
  const int badStarts = invalidStart.load(std::memory_order_relaxed);

  std::cerr << "datagen: done. wrote " << wrote << " samples to " << *outPath << " (games " << games << ", threads " << threads << ")\n";
  if (badStarts > 0) std::cerr << "datagen: warning: " << badStarts << " invalid start FENs skipped\n";
}

static std::string toLowerCopy(std::string_view sv) {
  std::string out;
  out.reserve(sv.size());
  for (const char ch : sv) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  return out;
}

static std::string normalizeToken(std::string_view sv) {
  std::string out;
  out.reserve(sv.size());
  for (const char ch : sv) {
    if (std::isspace(static_cast<unsigned char>(ch))) continue;
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return out;
}

static std::string moveToUciToken(const Move& m) {
  return normalizeToken(citadel::moveToString(m));
}

static std::string uciBestmoveToken(const Move& m) {
  if (m.to == citadel::SQ_NONE) return "0000";
  return moveToUciToken(m);
}

static std::optional<Move> parseMoveToken(Position& pos, std::string_view tok) {
  const std::string want = normalizeToken(tok);
  if (want.empty()) return std::nullopt;

  MoveList moves;
  pos.generateMoves(moves);
  for (std::uint32_t i = 0; i < moves.size; ++i) {
    if (moveToUciToken(moves.buf[i]) == want) return moves.buf[i];
  }
  // Also accept the "PGN token" spelling (same as moveToString but whitespace-free).
  for (std::uint32_t i = 0; i < moves.size; ++i) {
    if (normalizeToken(citadel::moveToPgnToken(moves.buf[i])) == want) return moves.buf[i];
  }

  return std::nullopt;
}

static std::string uciInfoLine(const citadel::SearchInfo& info) {
  constexpr int MATE_SCORE = 100'000'000;

  std::ostringstream oss;
  oss << "info depth " << info.depth;
  if (info.seldepth > 0) oss << " seldepth " << info.seldepth;

  oss << " score ";
  const int score = info.score;
  if (score > MATE_SCORE - 10'000 || score < -MATE_SCORE + 10'000) {
    const int matePlies = (score > 0) ? (MATE_SCORE - score) : (MATE_SCORE + score);
    const int mateMoves = (matePlies + 1) / 2;
    oss << "mate " << ((score > 0) ? mateMoves : -mateMoves);
  } else {
    oss << "cp " << score;
  }

  oss << " nodes " << info.nodes;
  const std::uint64_t nps = (info.timeMs > 0) ? (info.nodes * 1000ull / info.timeMs) : 0;
  oss << " nps " << nps;
  oss << " time " << info.timeMs;

  if (!info.pv.empty()) {
    oss << " pv";
    for (const auto& m : info.pv) oss << ' ' << moveToUciToken(m);
  }

  return oss.str();
}

enum class Classification { Best, Excellent, Okay, Inaccuracy, Mistake, Blunder };

static std::string classificationName(Classification c) {
  switch (c) {
    case Classification::Best: return "Best";
    case Classification::Excellent: return "Excellent";
    case Classification::Okay: return "Okay";
    case Classification::Inaccuracy: return "Inaccuracy";
    case Classification::Mistake: return "Mistake";
    case Classification::Blunder: return "Blunder";
  }
  return "???";
}

static std::string readAll(std::istream& is) {
  std::ostringstream oss;
  oss << is.rdbuf();
  return oss.str();
}

static std::optional<std::string> pgnTagValue(std::string_view pgn, std::string_view tag) {
  const std::string needle = "[" + std::string(tag) + " \"";
  const std::size_t at = std::string(pgn).find(needle);
  if (at == std::string::npos) return std::nullopt;
  std::size_t i = at + needle.size();

  std::string out;
  out.reserve(128);
  bool esc = false;
  while (i < pgn.size()) {
    const char ch = pgn[i++];
    if (esc) {
      out.push_back(ch);
      esc = false;
      continue;
    }
    if (ch == '\\') {
      esc = true;
      continue;
    }
    if (ch == '"') break;
    out.push_back(ch);
  }
  return out;
}

static bool isResultToken(std::string_view tok) {
  return tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*";
}

static bool isMoveNumberToken(std::string_view tok) {
  // Matches "12.", "12...", etc.
  std::size_t i = 0;
  while (i < tok.size() && std::isdigit(static_cast<unsigned char>(tok[i]))) ++i;
  if (i == 0 || i == tok.size()) return false;
  for (; i < tok.size(); ++i) {
    if (tok[i] != '.') return false;
  }
  return true;
}

static std::string stripTrailingAnnotations(std::string_view tok) {
  std::string s(tok);
  while (!s.empty()) {
    const char ch = s.back();
    if (ch == '!' || ch == '?' || ch == '+' || ch == '#') s.pop_back();
    else break;
  }
  return s;
}

static std::vector<std::string> pgnMoveTokens(std::string_view pgn) {
  // Strip tags, comments, and variations; then split into tokens.
  std::string cleaned;
  cleaned.reserve(pgn.size());
  bool inTag = false;
  bool inBrace = false;
  bool inSemicolon = false;
  int parenDepth = 0;

  for (std::size_t i = 0; i < pgn.size(); ++i) {
    const char ch = pgn[i];

    if (inSemicolon) {
      if (ch == '\n' || ch == '\r') inSemicolon = false;
      continue;
    }
    if (inTag) {
      if (ch == ']') inTag = false;
      continue;
    }
    if (inBrace) {
      if (ch == '}') inBrace = false;
      continue;
    }
    if (parenDepth > 0) {
      if (ch == '(') ++parenDepth;
      else if (ch == ')') --parenDepth;
      continue;
    }

    if (ch == '[') {
      inTag = true;
      continue;
    }
    if (ch == '{') {
      inBrace = true;
      continue;
    }
    if (ch == ';') {
      inSemicolon = true;
      continue;
    }
    if (ch == '(') {
      parenDepth = 1;
      continue;
    }

    cleaned.push_back(ch);
  }

  std::vector<std::string> out;
  std::string cur;
  cur.reserve(16);
  bool done = false;

  auto flush = [&]() {
    if (cur.empty() || done) return;
    // Ignore NAGs like "$1".
    if (cur.size() >= 2 && cur[0] == '$' &&
        std::all_of(cur.begin() + 1, cur.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; })) {
      cur.clear();
      return;
    }

    if (isMoveNumberToken(cur)) {
      cur.clear();
      return;
    }
    if (isResultToken(cur)) {
      done = true;
      cur.clear();
      return;
    }

    std::string tok = stripTrailingAnnotations(cur);
    if (!tok.empty()) out.push_back(std::move(tok));
    cur.clear();
  };

  for (const char ch : cleaned) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      flush();
      continue;
    }
    cur.push_back(ch);
  }
  flush();

  return out;
}

static constexpr int REVIEW_CP_FLOOR = 200;

static double reviewQuality01(int bestScore, int playedScore) {
  // Avoid pathological ratios when
  // `bestScore` is near 0 (e.g. best=16cp, played=2cp shouldn't be a blunder).
  //
  // We interpret "within X% of the best centipawns" as:
  //   quality = 1 - (centipawn_loss / max(|bestScore|, REVIEW_CP_FLOOR))
  //
  // This makes (in near-equal positions):
  //   loss 20cp -> quality 0.90 (Excellent)
  //   loss 40cp -> quality 0.80 (Okay)
  if (playedScore >= bestScore) return 1.0;
  const long long loss = static_cast<long long>(bestScore) - static_cast<long long>(playedScore); // >0
  const long long denom = std::max<long long>(std::llabs(static_cast<long long>(bestScore)), REVIEW_CP_FLOOR);
  double q = 1.0 - (static_cast<double>(loss) / static_cast<double>(denom));
  if (q < 0.0) q = 0.0;
  if (q > 1.0) q = 1.0;
  return q;
}

static void cmdReview(int argc, char** argv) {
  const int depth = intArg(argc, argv, "--depth", 4);
  const auto pgnPath = argValue(argc, argv, "--pgn");
  EvalContext ec = loadEvalForCommand(argc, argv);

  std::string pgn;
  if (!pgnPath || *pgnPath == "-") {
    if (!pgnPath) {
      std::cerr << "Paste PGN below, then press Ctrl-D:\n";
      std::cerr.flush();
    }
    pgn = readAll(std::cin);
  } else {
    std::ifstream f(*pgnPath);
    if (!f) throw std::runtime_error("Failed to open PGN file: " + *pgnPath);
    pgn = readAll(f);
  }

  if (pgn.empty()) throw std::runtime_error("review: empty PGN input");

  std::vector<std::string> moveTokens = pgnMoveTokens(pgn);
  Position pos = Position::initial();
  if (auto fen = pgnTagValue(pgn, "FEN")) pos = Position::fromFEN(*fen);

  std::cout << "Starting Game Review (depth " << depth << ")\n";
  std::cout << "------------------------------------------\n";

  for (std::size_t i = 0; i < moveTokens.size(); ++i) {
    const std::string& tok = moveTokens[i];
    const citadel::Color us = pos.turn();

    // IMPORTANT: run analysis on a copy so searching can't mutate the replay position.
    Position analysisPos = pos;
    citadel::SearchOptions opt;
    opt.limits.depth = depth;
    opt.evalBackend = ec.backend;
    opt.nnue = ec.nnuePtr();
    const auto r = citadel::searchBestMove(analysisPos, opt);
    const int bestScore = r.score;
    const Move bestMove = r.best;

    // Detect an *immediate* win (Regicide/Entombment) by testing the engine's best move.
    bool bestImmediateWin = false;
    citadel::WinReason bestImmediateReason = citadel::WinReason::None;
    {
      Position tmp = pos;
      citadel::Undo tu;
      tmp.makeMove(bestMove, tu);
      bestImmediateWin = tmp.gameOver() && tmp.winner().has_value() && *tmp.winner() == us &&
                         (tmp.winReason() == citadel::WinReason::Regicide || tmp.winReason() == citadel::WinReason::Entombment);
      bestImmediateReason = tmp.winReason();
    }

    const auto m = parseMoveToken(pos, tok);
    if (!m) {
      std::cout << "Ply " << (i + 1) << ": Failed to parse move token '" << tok << "'. Skipping rest.\n";
      break;
    }

    const std::string playedNorm = normalizeToken(tok);
    const std::string bestNorm = normalizeToken(citadel::moveToPgnToken(bestMove));
    const bool playedIsBest = (playedNorm == bestNorm);

    citadel::Undo u;
    pos.makeMove(*m, u);

    int playedScore = 0;
    const bool playedImmediateWin = pos.gameOver() && pos.winner().has_value() && *pos.winner() == us &&
                                   (pos.winReason() == citadel::WinReason::Regicide || pos.winReason() == citadel::WinReason::Entombment);

    if (playedImmediateWin) {
      playedScore = 99'999'999; // mateScore(1) equivalent
    } else if (playedIsBest) {
      playedScore = bestScore;
    } else {
      // Keep evaluation depth consistent with `bestScore`:
      // `bestScore` searches `depth` plies from the current position, so after forcing a move
      // we should search `depth-1` plies from the reply position (opponent to move).
      const int replyDepth = (depth > 1) ? (depth - 1) : 1;
      Position replyPos = pos;
      citadel::SearchOptions opt2;
      opt2.limits.depth = replyDepth;
      opt2.evalBackend = ec.backend;
      opt2.nnue = ec.nnuePtr();
      const auto r2 = citadel::searchBestMove(replyPos, opt2);
      playedScore = -r2.score; // convert back to mover's perspective
    }

    const bool missedImmediateWin = bestImmediateWin && !playedImmediateWin;

    Classification cl = Classification::Best;
    if (missedImmediateWin) {
      // Override: missed Regicide/Entombment.
      if (playedScore > 500) cl = Classification::Inaccuracy;
      else if (playedScore > 0) cl = Classification::Mistake;
      else cl = Classification::Blunder;
    } else if (playedImmediateWin || playedIsBest) {
      cl = Classification::Best;
    } else {
      const double q = reviewQuality01(bestScore, playedScore);
      if (q >= 0.90) cl = Classification::Excellent;
      else if (q >= 0.70) cl = Classification::Okay;
      else if (q >= 0.55) cl = Classification::Inaccuracy;
      else if (q >= 0.35) cl = Classification::Mistake;
      else cl = Classification::Blunder;
    }

    std::cout << std::setw(3) << (i + 1) << ". "
              << std::left << std::setw(5) << citadel::colorName(us) << ' '
              << std::left << std::setw(12) << tok
              << " | Eval: " << std::right << std::setw(6) << playedScore
              << " | Best: " << std::left << std::setw(12) << citadel::moveToPgnToken(bestMove) << " (" << std::right << std::setw(6) << bestScore << ")"
              << " | " << classificationName(cl);
    if (missedImmediateWin) {
      std::cout << " (missed " << ((bestImmediateReason == citadel::WinReason::Regicide) ? "Regicide" : "Entombment") << ")";
    }
    std::cout << "\n";
  }
}

static void uciLoop() {
  Position pos = Position::initial();

  citadel::EvalBackend evalBackend = citadel::EvalBackend::NNUE;
  citadel::NNUE nnue;
  std::string nnueFile;

  std::atomic_bool stop{false};
  std::thread worker;
  std::mutex outMu;

  auto send = [&](const std::string& s) {
    std::lock_guard<std::mutex> lk(outMu);
    std::cout << s << "\n";
    std::cout.flush();
  };

  auto stopSearch = [&]() {
    stop.store(true, std::memory_order_relaxed);
    if (worker.joinable()) worker.join();
    stop.store(false, std::memory_order_relaxed);
  };

  nnueFile = DEFAULT_NNUE_FILE;
  if (!nnue.loadFromFile(nnueFile)) {
    send("info string warning: nnue could not be loaded, falling back to HCE evaluation");
    evalBackend = citadel::EvalBackend::HCE;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    std::string_view sv = line;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) sv.remove_prefix(1);
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) sv.remove_suffix(1);
    if (sv.empty()) continue;

    std::istringstream iss{std::string(sv)};
    std::string cmd;
    iss >> cmd;
    cmd = toLowerCopy(cmd);

    if (cmd == "uci") {
      // Minimal UCI identification + a Hash option.
      send("id name Obelisk 0.1");
      send("id author Oscar");
      send("option name Hash type spin default " + std::to_string(citadel::transpositionTableSizeMB()) + " min 1 max 1024");
      send("option name Threads type spin default 1 min 1 max 1");
      send("option name Eval type combo default NNUE var HCE var NNUE");
      send(std::string("option name NnueFile type string default ") + DEFAULT_NNUE_FILE);
      send("uciok");
      continue;
    }

    if (cmd == "isready") {
      send("readyok");
      continue;
    }

    if (cmd == "ucinewgame") {
      stopSearch();
      citadel::clearTranspositionTable();
      pos = Position::initial();
      continue;
    }

    if (cmd == "setoption") {
      std::string tok;
      if (!(iss >> tok) || toLowerCopy(tok) != "name") continue;

      std::string name;
      while (iss >> tok) {
        if (toLowerCopy(tok) == "value") break;
        if (!name.empty()) name.push_back(' ');
        name += tok;
      }

      std::string value;
      {
        std::string vtok;
        while (iss >> vtok) {
          if (!value.empty()) value.push_back(' ');
          value += vtok;
        }
      }

      const std::string nameLower = toLowerCopy(name);
      if (nameLower == "hash" && !value.empty()) {
        const int mb = std::atoi(value.c_str());
        if (mb > 0) {
          stopSearch();
          citadel::setTranspositionTableSizeMB(static_cast<std::size_t>(mb));
        }
      }
      if (nameLower == "eval") {
        stopSearch();
        const std::string v = toLowerCopy(value);
        if (v == "nnue") {
          evalBackend = citadel::EvalBackend::NNUE;
          if (!nnue.loaded()) send("info string nnue not loaded (setoption name NnueFile value <path>)");
        } else {
          evalBackend = citadel::EvalBackend::HCE;
        }
      }
      if (nameLower == "nnuefile") {
        stopSearch();
        const std::string v = value;
        if (v.empty() || toLowerCopy(v) == "<empty>") {
          nnueFile.clear();
          send("info string nnue cleared");
        } else {
          nnueFile = v;
          if (!nnue.loadFromFile(nnueFile)) {
            send("info string nnue load failed: " + nnue.lastError());
          } else {
            send("info string nnue loaded: " + nnueFile);
          }
        }
      }
      continue;
    }

    if (cmd == "position") {
      stopSearch();

      std::string tok;
      if (!(iss >> tok)) continue;
      tok = toLowerCopy(tok);

      bool hasMoves = false;
      try {
        if (tok == "startpos") {
          pos = Position::initial();
          if (iss >> tok) {
            if (toLowerCopy(tok) == "moves") hasMoves = true;
          }
        } else if (tok == "fen") {
          std::vector<std::string> fenParts;
          fenParts.reserve(8);
          while (iss >> tok) {
            if (toLowerCopy(tok) == "moves") {
              hasMoves = true;
              break;
            }
            fenParts.push_back(tok);
          }
          std::string fen;
          for (std::size_t i = 0; i < fenParts.size(); ++i) {
            if (i) fen.push_back(' ');
            fen += fenParts[i];
          }
          pos = Position::fromFEN(fen);
        } else {
          continue;
        }

        if (hasMoves) {
          std::string mTok;
          while (iss >> mTok) {
            const auto m = parseMoveToken(pos, mTok);
            if (!m) {
              send("info string illegal move " + mTok);
              break;
            }
            citadel::Undo u;
            pos.makeMove(*m, u);
          }
        }
      } catch (const std::exception& e) {
        send(std::string("info string position error: ") + e.what());
      }
      continue;
    }

    // Non-standard but common debug helper: evaluate current position without searching.
    if (cmd == "eval") {
      citadel::EvalBackend eb = evalBackend;
      std::string tok;
      if (iss >> tok) {
        const std::string t = toLowerCopy(tok);
        if (t == "hce") eb = citadel::EvalBackend::HCE;
        if (t == "nnue") eb = citadel::EvalBackend::NNUE;
      }
      const citadel::NNUE* nn = (eb == citadel::EvalBackend::NNUE && nnue.loaded()) ? &nnue : nullptr;
      if (eb == citadel::EvalBackend::NNUE && !nn) {
        send("info string eval: nnue not loaded (setoption name NnueFile value <path>)");
      } else {
        const int score = citadel::evaluatePositionStm(pos, eb, nn);
        send(std::string("info string eval ") + ((eb == citadel::EvalBackend::NNUE) ? "NNUE" : "HCE") + " cp " + std::to_string(score));
      }
      continue;
    }

    if (cmd == "go") {
      stopSearch();

      int depth = 0;
      std::uint64_t movetimeMs = 0;
      std::uint64_t nodeLimit = 0;
      bool infinite = false;

      std::uint64_t wtime = 0, btime = 0, winc = 0, binc = 0;
      bool haveWtime = false, haveBtime = false, haveWinc = false, haveBinc = false;

      std::string tok;
      while (iss >> tok) {
        const std::string k = toLowerCopy(tok);
        if (k == "depth") {
          iss >> depth;
        } else if (k == "movetime") {
          iss >> movetimeMs;
        } else if (k == "nodes") {
          iss >> nodeLimit;
        } else if (k == "infinite") {
          infinite = true;
        } else if (k == "wtime") {
          iss >> wtime;
          haveWtime = true;
        } else if (k == "btime") {
          iss >> btime;
          haveBtime = true;
        } else if (k == "winc") {
          iss >> winc;
          haveWinc = true;
        } else if (k == "binc") {
          iss >> binc;
          haveBinc = true;
        } else {
          // ignore: ponder, mate, movestogo, etc.
        }
      }

      citadel::SearchLimits lim;
      lim.nodeLimit = nodeLimit;

      if (infinite) {
        lim.depth = 255;
        lim.timeLimitMs = 0;
      } else {
        lim.depth = (depth > 0) ? depth : 6;

        if (movetimeMs != 0) {
          lim.timeLimitMs = movetimeMs;
        } else if (haveWtime || haveBtime) {
          const std::uint64_t remaining = (pos.turn() == citadel::Color::White) ? (haveWtime ? wtime : 0) : (haveBtime ? btime : 0);
          const std::uint64_t inc = (pos.turn() == citadel::Color::White) ? (haveWinc ? winc : 0) : (haveBinc ? binc : 0);

          // Very simple time management: spend ~1/30th of remaining + half the increment.
          std::uint64_t budget = (remaining / 30) + (inc / 2);
          if (budget < 10) budget = 10;
          if (remaining > 50 && budget > remaining - 50) budget = remaining - 50;
          lim.timeLimitMs = budget;
        } else {
          lim.timeLimitMs = 0;
        }
      }

      stop.store(false, std::memory_order_relaxed);

      const citadel::EvalBackend evalForSearch = evalBackend;
      worker = std::thread([&, lim, evalForSearch]() {
        citadel::SearchOptions opt;
        opt.limits = lim;
        opt.stop = &stop;
        opt.onInfo = [&](const citadel::SearchInfo& info) { send(uciInfoLine(info)); };
        opt.evalBackend = evalForSearch;
        opt.nnue = (evalForSearch == citadel::EvalBackend::NNUE && nnue.loaded()) ? &nnue : nullptr;

        const auto r = citadel::searchBestMove(pos, opt);
        send("bestmove " + uciBestmoveToken(r.best));
      });

      continue;
    }

    if (cmd == "stop") {
      stopSearch();
      continue;
    }

    if (cmd == "quit") {
      stopSearch();
      break;
    }

    // Common debug convenience used by some GUIs / users.
    if (cmd == "d") {
      send(std::string("info string ") + pos.toFEN());
      continue;
    }
  }

  stopSearch();
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      uciLoop();
      return 0;
    }
    const std::string_view cmd = argv[1];
    if (cmd == "uci") {
      uciLoop();
      return 0;
    }
    if (cmd == "perft") {
      cmdPerft(argc, argv);
      return 0;
    }
    if (cmd == "bestmove") {
      cmdBestmove(argc, argv);
      return 0;
    }
    if (cmd == "play") {
      cmdPlay(argc, argv);
      return 0;
    }
    if (cmd == "selfplay") {
      cmdSelfplay(argc, argv);
      return 0;
    }
    if (cmd == "datagen") {
      cmdDatagen(argc, argv);
      return 0;
    }
    if (cmd == "review") {
      cmdReview(argc, argv);
      return 0;
    }

    usage(argv[0]);
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}

