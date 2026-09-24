// The runtime core, v0: device, buffers, runtime-compiled libraries, pipelines and timed command buffers of
// dispatches. This is what the bench harness and the kernel tests drive today and what the ICB builder, the
// host pump and the token ring build on (plan M2). Objective-C++ behind a C++ interface so the nanobind layer
// stays free of Objective-C.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace monolith {

struct DeviceImpl;
struct BufferImpl;
struct LibraryImpl;
struct PipelineImpl;
struct QueueImpl;

struct DeviceInfo {
  std::string name;
  int gpu_cores = 0;          // IORegistry gpu-core-count
  int apple_family = 0;       // highest supported MTLGPUFamilyApple<N>
  uint64_t max_buffer_length = 0;
  uint64_t recommended_working_set = 0;
  bool has_unified_memory = false;
};

class Device {
 public:
  Device();
  ~Device();
  DeviceInfo info() const;
  std::shared_ptr<DeviceImpl> impl;
};

class Buffer {
 public:
  // shared-storage buffer of `nbytes`, optionally initialized from `data`
  Buffer(const Device& d, size_t nbytes, const void* data = nullptr);
  // wraps a page-aligned range of a memory-mapped file with newBufferWithBytesNoCopy (no copy, no ownership)
  Buffer(const Device& d, const std::string& path, uint64_t offset, size_t nbytes);
  ~Buffer();
  size_t nbytes() const;
  void* contents() const;       // host pointer (shared storage / mmap)
  uint64_t gpu_address() const;
  std::shared_ptr<BufferImpl> impl;
};

class Library {
 public:
  // compiles MSL source at run time; `macros` become -D definitions; language_version e.g. 0x30002 (3.2), 0x40000
  Library(const Device& d, const std::string& source, const std::map<std::string, std::string>& macros,
          uint32_t language_version, bool fast_math);
  ~Library();
  std::shared_ptr<LibraryImpl> impl;
};

class Pipeline {
 public:
  Pipeline(const Library& lib, const std::string& function, bool support_icb);
  ~Pipeline();
  uint32_t max_threads_per_threadgroup() const;
  uint32_t thread_execution_width() const;
  std::shared_ptr<PipelineImpl> impl;
};

struct BufferBinding { uint32_t index; const Buffer* buffer; uint64_t offset; };
struct BytesBinding { uint32_t index; std::vector<uint8_t> bytes; };

struct Dispatch {
  const Pipeline* pipeline = nullptr;
  std::vector<BufferBinding> buffers;
  std::vector<BytesBinding> bytes;
  std::vector<std::pair<uint32_t, uint32_t>> threadgroup_memory;   // (index, length)
  uint32_t grid[3] = {1, 1, 1};        // threadgroups
  uint32_t threadgroup[3] = {1, 1, 1}; // threads per threadgroup
  bool barrier_after = false;          // concurrent encoders only
};

struct RunResult { double gpu_ms; double wall_ms; std::string error; };

class Queue {
 public:
  explicit Queue(const Device& d);
  ~Queue();
  // one command buffer holding `dispatches` in order (serial encoder, or concurrent with explicit barriers),
  // committed and waited for; returns GPU time from the command buffer timestamps
  RunResult run(const std::vector<Dispatch>& dispatches, bool concurrent);
  std::shared_ptr<QueueImpl> impl;
};

}  // namespace monolith
