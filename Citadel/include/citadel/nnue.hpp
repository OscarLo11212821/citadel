#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "citadel/core.hpp"
#include "citadel/position.hpp"

namespace citadel {

// A small, quantized NNUE-style evaluator:
// - Sparse feature transform (piece/wall-on-square + a few global bits) summed into an accumulator.
// - A small MLP head.
//
// The model is trained with quantization-aware training (QAT) and exported to a compact binary
// file that this class can load.
class NNUE {
public:
  static constexpr std::uint32_t kVersion = 1;

  // Input features:
  // - 16 channels per square (white pieces 0..5, white walls 6..7, black pieces 8..13, black walls 14..15)
  // - plus 3 global bits: white-to-move, white-bastion-right, black-bastion-right
  static constexpr std::uint32_t kBoardChannels = 16;
  static constexpr std::uint32_t kGlobalFeatures = 3;
  static constexpr std::uint32_t kInputDim = kBoardChannels * static_cast<std::uint32_t>(SQ_N) + kGlobalFeatures;

  // Network sizes (fixed for simplicity).
  static constexpr std::uint32_t kHidden1 = 256;
  static constexpr std::uint32_t kHidden2 = 32;

  // Clipped ReLU range (0..kActMax).
  static constexpr std::uint32_t kActMax = 127;

  struct Accumulator {
    std::array<std::int32_t, kHidden1> v{};
  };

  [[nodiscard]] bool loaded() const { return loaded_; }
  [[nodiscard]] const std::string& lastError() const { return lastError_; }

  // Load a quantized model from disk.
  [[nodiscard]] bool loadFromFile(const std::string& path);

  // Evaluate from side-to-move perspective (positive = good for side to move).
  [[nodiscard]] int evaluateStm(const Position& pos, const Accumulator& acc) const;

  // Build an accumulator from scratch for the given position.
  void initAccumulator(const Position& pos, Accumulator& out) const;

  // Update an accumulator after applying a normal move to the position.
  // `posAfterMove` must be the position AFTER `makeMove(m, u)`.
  void applyDeltaAfterMove(Accumulator& acc, const Position& posAfterMove, const Undo& u) const;

  // Update an accumulator after applying a null move to the position.
  // `posAfterNull` must be the position AFTER `makeNullMove(u)`.
  void applyDeltaAfterNullMove(Accumulator& acc, const Position& posAfterNull, const NullUndo& u) const;

private:
  // Feature indices for globals.
  static constexpr std::uint32_t kFeatStmWhite = kBoardChannels * static_cast<std::uint32_t>(SQ_N) + 0;
  static constexpr std::uint32_t kFeatBastionWhite = kBoardChannels * static_cast<std::uint32_t>(SQ_N) + 1;
  static constexpr std::uint32_t kFeatBastionBlack = kBoardChannels * static_cast<std::uint32_t>(SQ_N) + 2;

  // Model parameters (quantized).
  // Feature transform weights are stored feature-major for fast incremental updates:
  //   ftW_[feature * kHidden1 + j] is the contribution to hidden unit j.
  std::array<std::int32_t, kHidden1> ftB_{};
  std::array<std::int32_t, kHidden2> l2B_{};
  std::int32_t outB_ = 0;

  std::array<std::int8_t, kHidden2> outW_{};
  std::vector<std::int16_t> ftW_; // kInputDim * kHidden1
  std::vector<std::int8_t> l2W_;  // kHidden2 * kHidden1

  std::uint32_t shift2_ = 12;
  std::uint32_t shift3_ = 8;

  bool loaded_ = false;
  std::string lastError_{};

  [[nodiscard]] static std::uint32_t featureIndex(std::uint8_t sq, std::int8_t raw);
  [[nodiscard]] static inline int arshift(int x, std::uint32_t s) {
    if (s == 0) return x;
    if (x >= 0) return x >> s;
    // Defined arithmetic shift: floor division for negatives.
    const int neg = -x;
    const int add = (1 << static_cast<int>(s)) - 1;
    return -((neg + add) >> s);
  }

  [[nodiscard]] int evaluateWhite(const Position& pos, const Accumulator& acc) const;
};

} // namespace citadel


