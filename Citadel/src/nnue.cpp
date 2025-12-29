#include "citadel/nnue.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <limits>

namespace citadel {

namespace {

static bool readExact(std::istream& in, void* dst, std::size_t n) {
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
  return static_cast<std::size_t>(in.gcount()) == n;
}

static bool readU32(std::istream& in, std::uint32_t& out) {
  return readExact(in, &out, sizeof(out));
}

static bool readI32(std::istream& in, std::int32_t& out) {
  return readExact(in, &out, sizeof(out));
}

static bool readI16(std::istream& in, std::int16_t& out) {
  return readExact(in, &out, sizeof(out));
}

static bool readI8(std::istream& in, std::int8_t& out) {
  return readExact(in, &out, sizeof(out));
}

} // namespace

std::uint32_t NNUE::featureIndex(std::uint8_t sq, std::int8_t raw) {
  if (sq >= SQ_N) return std::numeric_limits<std::uint32_t>::max();
  if (raw == 0) return std::numeric_limits<std::uint32_t>::max();

  const bool isWhite = raw > 0;
  const int av = (raw < 0) ? -static_cast<int>(raw) : static_cast<int>(raw);

  std::uint32_t ch = 0;
  if (av >= 1 && av <= 6) {
    // Piece.
    const std::uint32_t pt = static_cast<std::uint32_t>(av - 1); // 0..5
    ch = (isWhite ? 0u : 8u) + pt;
  } else if (av == 7) {
    // Wall hp1.
    ch = isWhite ? 6u : 14u;
  } else if (av == 8) {
    // Wall hp2.
    ch = isWhite ? 7u : 15u;
  } else {
    return std::numeric_limits<std::uint32_t>::max();
  }

  return static_cast<std::uint32_t>(sq) * kBoardChannels + ch;
}

bool NNUE::loadFromFile(const std::string& path) {
  loaded_ = false;
  lastError_.clear();
  ftW_.clear();
  l2W_.clear();

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    lastError_ = "NNUE: failed to open file";
    return false;
  }

  char magic[4]{};
  if (!readExact(in, magic, sizeof(magic))) {
    lastError_ = "NNUE: failed to read header";
    return false;
  }
  if (!(magic[0] == 'C' && magic[1] == 'N' && magic[2] == 'U' && magic[3] == 'E')) {
    lastError_ = "NNUE: bad magic (expected CNUE)";
    return false;
  }

  std::uint32_t version = 0;
  std::uint32_t inputDim = 0;
  std::uint32_t h1 = 0;
  std::uint32_t h2 = 0;
  std::uint32_t actMax = 0;
  std::uint32_t shift2 = 0;
  std::uint32_t shift3 = 0;
  if (!readU32(in, version) || !readU32(in, inputDim) || !readU32(in, h1) || !readU32(in, h2) || !readU32(in, actMax) || !readU32(in, shift2) ||
      !readU32(in, shift3)) {
    lastError_ = "NNUE: failed to read header fields";
    return false;
  }

  if (version != kVersion) {
    lastError_ = "NNUE: unsupported version";
    return false;
  }
  if (inputDim != kInputDim || h1 != kHidden1 || h2 != kHidden2) {
    lastError_ = "NNUE: shape mismatch (model vs engine)";
    return false;
  }
  if (actMax != kActMax) {
    lastError_ = "NNUE: activation clamp mismatch";
    return false;
  }
  if (shift2 > 31 || shift3 > 31) {
    lastError_ = "NNUE: invalid shift values";
    return false;
  }

  shift2_ = shift2;
  shift3_ = shift3;

  // Feature-transform weights/bias.
  ftW_.resize(static_cast<std::size_t>(kInputDim) * kHidden1);
  for (std::size_t i = 0; i < ftW_.size(); ++i) {
    std::int16_t v = 0;
    if (!readI16(in, v)) {
      lastError_ = "NNUE: failed to read ftW";
      return false;
    }
    ftW_[i] = v;
  }
  for (std::size_t j = 0; j < kHidden1; ++j) {
    std::int32_t v = 0;
    if (!readI32(in, v)) {
      lastError_ = "NNUE: failed to read ftB";
      return false;
    }
    ftB_[j] = v;
  }

  // Layer 2.
  l2W_.resize(static_cast<std::size_t>(kHidden2) * kHidden1);
  for (std::size_t i = 0; i < l2W_.size(); ++i) {
    std::int8_t v = 0;
    if (!readI8(in, v)) {
      lastError_ = "NNUE: failed to read l2W";
      return false;
    }
    l2W_[i] = v;
  }
  for (std::size_t j = 0; j < kHidden2; ++j) {
    std::int32_t v = 0;
    if (!readI32(in, v)) {
      lastError_ = "NNUE: failed to read l2B";
      return false;
    }
    l2B_[j] = v;
  }

  // Output.
  for (std::size_t j = 0; j < kHidden2; ++j) {
    std::int8_t v = 0;
    if (!readI8(in, v)) {
      lastError_ = "NNUE: failed to read outW";
      return false;
    }
    outW_[j] = v;
  }
  {
    std::int32_t v = 0;
    if (!readI32(in, v)) {
      lastError_ = "NNUE: failed to read outB";
      return false;
    }
    outB_ = v;
  }

  loaded_ = true;
  return true;
}

void NNUE::initAccumulator(const Position& pos, Accumulator& out) const {
  // Start with bias.
  for (std::size_t j = 0; j < kHidden1; ++j) out.v[j] = ftB_[j];

  // Board features.
  for (std::uint8_t s = 0; s < SQ_N; ++s) {
    const std::int8_t raw = pos.rawAt(s);
    const std::uint32_t f = featureIndex(s, raw);
    if (f == std::numeric_limits<std::uint32_t>::max()) continue;
    const std::size_t base = static_cast<std::size_t>(f) * kHidden1;
    for (std::size_t j = 0; j < kHidden1; ++j) out.v[j] += ftW_[base + j];
  }

  // Global bits.
  auto addBit = [&](std::uint32_t feat, bool on) {
    if (!on) return;
    const std::size_t base = static_cast<std::size_t>(feat) * kHidden1;
    for (std::size_t j = 0; j < kHidden1; ++j) out.v[j] += ftW_[base + j];
  };
  addBit(kFeatStmWhite, pos.turn() == Color::White);
  addBit(kFeatBastionWhite, pos.bastionRight(Color::White));
  addBit(kFeatBastionBlack, pos.bastionRight(Color::Black));
}

void NNUE::applyDeltaAfterMove(Accumulator& acc, const Position& posAfterMove, const Undo& u) const {
  // Changed squares.
  for (std::uint8_t i = 0; i < u.sqCount; ++i) {
    const std::uint8_t s = u.sq[i];
    const std::int8_t oldRaw = u.prev[i];
    const std::int8_t newRaw = posAfterMove.rawAt(s);

    if (oldRaw != 0) {
      const std::uint32_t fOld = featureIndex(s, oldRaw);
      if (fOld != std::numeric_limits<std::uint32_t>::max()) {
        const std::size_t base = static_cast<std::size_t>(fOld) * kHidden1;
        for (std::size_t j = 0; j < kHidden1; ++j) acc.v[j] -= ftW_[base + j];
      }
    }
    if (newRaw != 0) {
      const std::uint32_t fNew = featureIndex(s, newRaw);
      if (fNew != std::numeric_limits<std::uint32_t>::max()) {
        const std::size_t base = static_cast<std::size_t>(fNew) * kHidden1;
        for (std::size_t j = 0; j < kHidden1; ++j) acc.v[j] += ftW_[base + j];
      }
    }
  }

  // Global bits (turn + bastion rights).
  const bool prevStmWhite = (u.prevTurn == Color::White);
  const bool newStmWhite = (posAfterMove.turn() == Color::White);
  if (prevStmWhite != newStmWhite) {
    const std::size_t base = static_cast<std::size_t>(kFeatStmWhite) * kHidden1;
    for (std::size_t j = 0; j < kHidden1; ++j) acc.v[j] += newStmWhite ? ftW_[base + j] : -ftW_[base + j];
  }

  const bool prevBW = u.prevBastionRight[static_cast<int>(Color::White)];
  const bool prevBB = u.prevBastionRight[static_cast<int>(Color::Black)];
  const bool newBW = posAfterMove.bastionRight(Color::White);
  const bool newBB = posAfterMove.bastionRight(Color::Black);

  if (prevBW != newBW) {
    const std::size_t base = static_cast<std::size_t>(kFeatBastionWhite) * kHidden1;
    for (std::size_t j = 0; j < kHidden1; ++j) acc.v[j] += newBW ? ftW_[base + j] : -ftW_[base + j];
  }
  if (prevBB != newBB) {
    const std::size_t base = static_cast<std::size_t>(kFeatBastionBlack) * kHidden1;
    for (std::size_t j = 0; j < kHidden1; ++j) acc.v[j] += newBB ? ftW_[base + j] : -ftW_[base + j];
  }
}

void NNUE::applyDeltaAfterNullMove(Accumulator& acc, const Position& posAfterNull, const NullUndo& u) const {
  const bool prevStmWhite = (u.prevTurn == Color::White);
  const bool newStmWhite = (posAfterNull.turn() == Color::White);
  if (prevStmWhite == newStmWhite) return;
  const std::size_t base = static_cast<std::size_t>(kFeatStmWhite) * kHidden1;
  for (std::size_t j = 0; j < kHidden1; ++j) acc.v[j] += newStmWhite ? ftW_[base + j] : -ftW_[base + j];
}

int NNUE::evaluateWhite(const Position& pos, const Accumulator& acc) const {
  (void)pos;
  // Hidden1 activations.
  std::array<std::uint8_t, kHidden1> h1{};
  for (std::size_t j = 0; j < kHidden1; ++j) {
    int x = acc.v[j];
    if (x < 0) x = 0;
    if (x > static_cast<int>(kActMax)) x = static_cast<int>(kActMax);
    h1[j] = static_cast<std::uint8_t>(x);
  }

  // Hidden2 activations.
  std::array<std::uint8_t, kHidden2> h2{};
  for (std::size_t k = 0; k < kHidden2; ++k) {
    std::int32_t sum = l2B_[k];
    const std::int8_t* w = &l2W_[k * kHidden1];
    for (std::size_t j = 0; j < kHidden1; ++j) sum += static_cast<std::int32_t>(w[j]) * static_cast<std::int32_t>(h1[j]);
    sum = static_cast<std::int32_t>(arshift(static_cast<int>(sum), shift2_));
    int x = static_cast<int>(sum);
    if (x < 0) x = 0;
    if (x > static_cast<int>(kActMax)) x = static_cast<int>(kActMax);
    h2[k] = static_cast<std::uint8_t>(x);
  }

  std::int32_t out = outB_;
  for (std::size_t k = 0; k < kHidden2; ++k) out += static_cast<std::int32_t>(outW_[k]) * static_cast<std::int32_t>(h2[k]);

  return arshift(static_cast<int>(out), shift3_);
}

int NNUE::evaluateStm(const Position& pos, const Accumulator& acc) const {
  const int scoreW = evaluateWhite(pos, acc);
  return (pos.turn() == Color::White) ? scoreW : -scoreW;
}

} // namespace citadel


