#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include "metal_core.h"

namespace monolith {

struct DeviceImpl { id<MTLDevice> dev; };
struct BufferImpl { id<MTLBuffer> buf; void* mapped = nullptr; size_t mapped_len = 0; };
struct LibraryImpl { id<MTLLibrary> lib; id<MTLDevice> dev; };
struct PipelineImpl { id<MTLComputePipelineState> pso; };
struct QueueImpl { id<MTLCommandQueue> q; };

static int ioreg_gpu_cores() {
  int n = 0; io_iterator_t it;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) == KERN_SUCCESS) {
    io_object_t o;
    while ((o = IOIteratorNext(it))) {
      CFTypeRef p = IORegistryEntryCreateCFProperty(o, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (p) { if (CFGetTypeID(p) == CFNumberGetTypeID()) CFNumberGetValue((CFNumberRef)p, kCFNumberIntType, &n); CFRelease(p); }
      IOObjectRelease(o); if (n > 0) break;
    }
    IOObjectRelease(it);
  }
  return n;
}

Device::Device() : impl(std::make_shared<DeviceImpl>()) {
  impl->dev = MTLCreateSystemDefaultDevice();
  if (!impl->dev) throw std::runtime_error("no Metal device");
}
Device::~Device() = default;

DeviceInfo Device::info() const {
  DeviceInfo i;
  i.name = [impl->dev.name UTF8String];
  i.gpu_cores = ioreg_gpu_cores();
  for (int f = 1; f <= 12; f++) if ([impl->dev supportsFamily:(MTLGPUFamily)(1000 + f)]) i.apple_family = f;
  i.max_buffer_length = impl->dev.maxBufferLength;
  i.recommended_working_set = impl->dev.recommendedMaxWorkingSetSize;
  i.has_unified_memory = impl->dev.hasUnifiedMemory;
  return i;
}

Buffer::Buffer(const Device& d, size_t nbytes, const void* data) : impl(std::make_shared<BufferImpl>()) {
  impl->buf = data ? [d.impl->dev newBufferWithBytes:data length:nbytes options:MTLResourceStorageModeShared]
                   : [d.impl->dev newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
  if (!impl->buf) throw std::runtime_error("newBuffer failed (" + std::to_string(nbytes) + " bytes)");
}

Buffer::Buffer(const Device& d, const std::string& path, uint64_t offset, size_t nbytes) : impl(std::make_shared<BufferImpl>()) {
  long page = sysconf(_SC_PAGESIZE);
  if (offset % page || nbytes % page) throw std::runtime_error("mmap buffers need page-aligned offset and length");
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("open failed: " + path);
  void* p = mmap(nullptr, nbytes, PROT_READ, MAP_PRIVATE, fd, (off_t)offset);
  close(fd);
  if (p == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  impl->mapped = p; impl->mapped_len = nbytes;
  impl->buf = [d.impl->dev newBufferWithBytesNoCopy:p length:nbytes options:MTLResourceStorageModeShared deallocator:nil];
  if (!impl->buf) { munmap(p, nbytes); throw std::runtime_error("newBufferWithBytesNoCopy failed"); }
}

Buffer::~Buffer() { if (impl && impl.use_count() == 1 && impl->mapped) { impl->buf = nil; munmap(impl->mapped, impl->mapped_len); } }
size_t Buffer::nbytes() const { return impl->buf.length; }
void* Buffer::contents() const { return impl->buf.contents; }
uint64_t Buffer::gpu_address() const { return impl->buf.gpuAddress; }

Library::Library(const Device& d, const std::string& source, const std::map<std::string, std::string>& macros,
                 uint32_t language_version, bool fast_math) : impl(std::make_shared<LibraryImpl>()) {
  MTLCompileOptions* o = [MTLCompileOptions new];
  if (language_version) o.languageVersion = (MTLLanguageVersion)language_version;
  o.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
  NSMutableDictionary* m = [NSMutableDictionary new];
  for (auto& kv : macros) m[[NSString stringWithUTF8String:kv.first.c_str()]] = [NSString stringWithUTF8String:kv.second.c_str()];
  o.preprocessorMacros = m;
  NSError* err = nil;
  impl->dev = d.impl->dev;
  impl->lib = [d.impl->dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()] options:o error:&err];
  if (!impl->lib) throw std::runtime_error(std::string("MSL compile failed: ") + (err ? [err.localizedDescription UTF8String] : "?"));
}
Library::~Library() = default;

Pipeline::Pipeline(const Library& lib, const std::string& function, bool support_icb) : impl(std::make_shared<PipelineImpl>()) {
  id<MTLFunction> fn = [lib.impl->lib newFunctionWithName:[NSString stringWithUTF8String:function.c_str()]];
  if (!fn) throw std::runtime_error("no kernel named " + function);
  NSError* err = nil;
  MTLComputePipelineDescriptor* pd = [MTLComputePipelineDescriptor new];
  pd.computeFunction = fn; pd.supportIndirectCommandBuffers = support_icb;
  impl->pso = [lib.impl->dev newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&err];
  if (!impl->pso) throw std::runtime_error(std::string("pipeline failed: ") + (err ? [err.localizedDescription UTF8String] : "?"));
}
Pipeline::~Pipeline() = default;
uint32_t Pipeline::max_threads_per_threadgroup() const { return (uint32_t)impl->pso.maxTotalThreadsPerThreadgroup; }
uint32_t Pipeline::thread_execution_width() const { return (uint32_t)impl->pso.threadExecutionWidth; }

Queue::Queue(const Device& d) : impl(std::make_shared<QueueImpl>()) { impl->q = [d.impl->dev newCommandQueue]; }
Queue::~Queue() = default;

static double now_ms() { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6; }

RunResult Queue::run(const std::vector<Dispatch>& dispatches, bool concurrent) {
  @autoreleasepool {
    double t0 = now_ms();
    id<MTLCommandBuffer> cb = [impl->q commandBuffer];
    id<MTLComputeCommandEncoder> en = concurrent ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] : [cb computeCommandEncoder];
    for (auto& d : dispatches) {
      [en setComputePipelineState:d.pipeline->impl->pso];
      for (auto& b : d.buffers) [en setBuffer:b.buffer->impl->buf offset:b.offset atIndex:b.index];
      for (auto& b : d.bytes) [en setBytes:b.bytes.data() length:b.bytes.size() atIndex:b.index];
      for (auto& t : d.threadgroup_memory) [en setThreadgroupMemoryLength:t.second atIndex:t.first];
      [en dispatchThreadgroups:MTLSizeMake(d.grid[0], d.grid[1], d.grid[2]) threadsPerThreadgroup:MTLSizeMake(d.threadgroup[0], d.threadgroup[1], d.threadgroup[2])];
      if (concurrent && d.barrier_after) [en memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    RunResult r;
    r.gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
    r.wall_ms = now_ms() - t0;
    if (cb.error) r.error = [cb.error.localizedDescription UTF8String];
    return r;
  }
}

}  // namespace monolith
