// Minimal audio input for examples, tests and benchmarks.
#pragma once

#include <string>
#include <vector>

// Reads a RIFF/WAVE file (PCM 8/16/24/32-bit, IEEE float 32/64; channels are
// averaged to mono) or a raw little-endian float32 mono file (*.f32, assumed to
// be at target_rate). Resamples to target_rate with a windowed-sinc filter when
// needed. Returns false and sets err on failure.
bool read_audio(const std::string & path, int target_rate, std::vector<float> & out, std::string & err);

// Reads a raw float32 file.
bool read_f32(const std::string & path, std::vector<float> & out);
