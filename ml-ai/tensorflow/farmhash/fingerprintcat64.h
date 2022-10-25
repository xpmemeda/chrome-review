#pragma once

typedef uint64_t uint64;

namespace internal {
// Mixes some of the bits that got propagated to the high bits back into the
// low bits.
inline uint64 ShiftMix(const uint64 val) { return val ^ (val >> 47); }
} // namespace internal

// This concatenates two 64-bit fingerprints. It is a convenience function to
// get a fingerprint for a combination of already fingerprinted components. For
// example this code is used to concatenate the hashes from each of the features
// on sparse crosses.
//
// One shouldn't expect FingerprintCat64(Fingerprint64(x), Fingerprint64(y))
// to indicate anything about FingerprintCat64(StrCat(x, y)). This operation
// is not commutative.
//
// From a security standpoint, we don't encourage this pattern to be used
// for everything as it is vulnerable to length-extension attacks and it
// is easier to compute multicollisions.
inline uint64 FingerprintCat64(const uint64 fp1, const uint64 fp2) {
  static const uint64 kMul = 0xc6a4a7935bd1e995ULL;
  uint64 result = fp1 ^ kMul;
  result ^= internal::ShiftMix(fp2 * kMul) * kMul;
  result *= kMul;
  result = internal::ShiftMix(result) * kMul;
  result = internal::ShiftMix(result);
  return result;
}