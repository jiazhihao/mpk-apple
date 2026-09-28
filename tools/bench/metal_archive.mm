// Compile an exported KernelSpec with the same public Metal APIs as the runtime.
// Build: clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
//          tools/bench/metal_archive.mm -o /tmp/metal-archive
// No Xcode Metal compiler/disassembler component is required.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main(int argc, const char **argv) {
 @autoreleasepool {
  if (argc != 3) { fprintf(stderr,"usage: archive spec.json output.metallib\n"); return 2; }
  NSError *err=nil;
  NSData *data=[NSData dataWithContentsOfFile:@(argv[1])];
  if (!data) { fprintf(stderr,"cannot read spec\n"); return 1; }
  NSDictionary *spec=[NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
  if (!spec) { NSLog(@"spec: %@",err); return 1; }
  id<MTLDevice> dev=MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr,"no Metal device\n"); return 1; }
  MTLCompileOptions *opts=[MTLCompileOptions new];
  opts.mathMode=[spec[@"fast_math"] boolValue] ? MTLMathModeFast : MTLMathModeSafe;
  if ([spec[@"language_version"] unsignedIntValue]) opts.languageVersion=(MTLLanguageVersion)[spec[@"language_version"] unsignedIntValue];
  opts.preprocessorMacros=spec[@"macros"];
  id<MTLLibrary> lib=[dev newLibraryWithSource:spec[@"source"] options:opts error:&err];
  if (!lib) { NSLog(@"compile: %@",err); return 1; }
  MTLComputePipelineDescriptor *pd=[MTLComputePipelineDescriptor new];
  pd.maxTotalThreadsPerThreadgroup=[spec[@"max_threads"] unsignedIntValue];
  pd.threadGroupSizeIsMultipleOfThreadExecutionWidth=[spec[@"simd_multiple"] boolValue];
  pd.computeFunction=[lib newFunctionWithName:spec[@"function"]];
  pd.supportIndirectCommandBuffers=spec[@"support_icb"] ? [spec[@"support_icb"] boolValue] : YES;
  if (!pd.computeFunction) { fprintf(stderr,"function missing\n"); return 1; }
  id<MTLComputePipelineState> pso=[dev newComputePipelineStateWithDescriptor:pd options:0 reflection:nil error:&err];
  if (!pso) { NSLog(@"pipeline: %@",err); return 1; }
  id<MTLBinaryArchive> archive=[dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&err];
  if (!archive || ![archive addComputePipelineFunctionsWithDescriptor:pd error:&err] || ![archive serializeToURL:[NSURL fileURLWithPath:@(argv[2])] error:&err]) { NSLog(@"archive: %@",err); return 1; }
  NSDictionary *result=@{@"device":dev.name,@"function":spec[@"function"],@"max_threads":@(pso.maxTotalThreadsPerThreadgroup),@"simd_width":@(pso.threadExecutionWidth),@"static_threadgroup_bytes":@(pso.staticThreadgroupMemoryLength)};
  NSData *out=[NSJSONSerialization dataWithJSONObject:result options:0 error:nil];
  fwrite(out.bytes,1,out.length,stdout); puts("");
 }
 return 0;
}
