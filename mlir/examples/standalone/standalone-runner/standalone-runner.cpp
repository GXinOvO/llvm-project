#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Error.h"

#include <iostream>
#include <string>

using namespace mlir;

static OwningOpRef<ModuleOp>
loadModule(llvm::StringRef filename, MLIRContext &context) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Error reading input file: " << ec.message() << "\n";
    return {};
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  SourceMgrDiagnosticHandler diagHandler(sourceMgr, &context);

  OwningOpRef<ModuleOp> module =
      parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error: failed to parse MLIR file\n";
  }
  return module;
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  if (argc < 3) {
    llvm::errs()
        << "Usage:\n"
        << "  " << argv[0]
        << " linear-int <llvm-mlir-file> x w b\n"
        << "  " << argv[0]
        << " linear-mem <llvm-mlir-file>\n";
    return 1;
  }

  std::string mode = argv[1];
  llvm::StringRef inputFilename = argv[2];

  // 1. 准备 MLIR Context，只注册 LLVM Dialect
  DialectRegistry registry;
  registry.insert<LLVM::LLVMDialect>();
  MLIRContext context(registry);

  // 2. 读取（已经是 LLVM Dialect 的）MLIR 模块
  OwningOpRef<ModuleOp> module = loadModule(inputFilename, context);
  if (!module)
    return 1;

  // 3. 注册 MLIR -> LLVM IR 的翻译
  registerBuiltinDialectTranslation(context);
  registerLLVMDialectTranslation(context);

  // 4. 初始化 JIT 目标（本机 CPU）
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // 5. 创建优化 pipeline（O2），只做 LLVM IR 优化，不做 Dialect 转换
  auto optPipeline =
      makeOptimizingTransformer(/*optLevel=*/2,
                                /*sizeLevel=*/0,
                                /*targetMachine=*/nullptr);

  ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;

  auto jitOrError = ExecutionEngine::create(*module, engineOptions);
  if (!jitOrError) {
    llvm::errs() << "Failed to create ExecutionEngine\n";
    llvm::logAllUnhandledErrors(jitOrError.takeError(), llvm::errs(),
                                "Error: ");
    return 1;
  }
  std::unique_ptr<ExecutionEngine> jit = std::move(*jitOrError);

  //===============================
  // 分两种模式处理
  //===============================
  if (mode == "linear-int") {
    //--------------------------------------
    // 整数版本：y = x * w + b
    //--------------------------------------
    if (argc < 6) {
      llvm::errs()
          << "Usage: " << argv[0]
          << " linear-int <llvm-mlir-file> x w b\n";
      return 1;
    }

    int x = std::stoi(argv[3]);
    int w = std::stoi(argv[4]);
    int b = std::stoi(argv[5]);

    // 查找函数 @test
    auto fPtrOrErr = jit->lookup("test");
    if (!fPtrOrErr) {
      llvm::errs() << "ERROR: function @test not found in module\n";
      llvm::logAllUnhandledErrors(fPtrOrErr.takeError(), llvm::errs(),
                                  "Error: ");
      return 1;
    }

    using FuncPtr = int (*)(int, int, int);
    FuncPtr fn = reinterpret_cast<FuncPtr>(*fPtrOrErr);

    int result = fn(x, w, b);
    std::cout << "Result = " << result << "\n";
    return 0;

  } else if (mode == "linear-mem") {
    //--------------------------------------
    // 矩阵版本：y = A * x
    //--------------------------------------
    // 构造 A, x, y 的数据
    //
    //   A: memref<4x3xf32>
    //   x: memref<3xf32>
    //   y: memref<4xf32>
    //
    //   A = [ [ 1,  2,  3],
    //         [ 4,  5,  6],
    //         [ 7,  8,  9],
    //         [10, 11, 12] ]
    //
    //   x = [1, 1, 1]
    //   y 初始全 0
    //--------------------------------------
    float AData[4 * 3];
    float xData[3];
    float yData[4];

    int idx = 0;
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 3; ++j) {
        AData[idx++] = static_cast<float>(i * 3 + j + 1); // 1..12
      }
    }

    for (int j = 0; j < 3; ++j)
      xData[j] = 1.0f;

    for (int i = 0; i < 4; ++i)
      yData[i] = 0.0f;

    // 构造 StridedMemRefType descriptor
    using MemRef2D = StridedMemRefType<float, 2>;
    using MemRef1D = StridedMemRefType<float, 1>;

    MemRef2D A;
    A.basePtr = AData;
    A.data = AData;
    A.offset = 0;
    A.sizes[0] = 4; // 行数
    A.sizes[1] = 3; // 列数
    A.strides[1] = 1; // 最后一维：连续存储
    A.strides[0] = 3; // 每行跨 3 个元素

    MemRef1D x;
    x.basePtr = xData;
    x.data = xData;
    x.offset = 0;
    x.sizes[0] = 3;
    x.strides[0] = 1;

    MemRef1D y;
    y.basePtr = yData;
    y.data = yData;
    y.offset = 0;
    y.sizes[0] = 4;
    y.strides[0] = 1;

    // 调用 @main(A, x, y)
    if (auto err = jit->invoke("main", &A, &x, &y)) {
      llvm::errs() << "JIT invocation failed\n";
      llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "Error: ");
      return 1;
    }

    // 打印 y 结果
    std::cout << "y = [";
    for (int i = 0; i < 4; ++i) {
      std::cout << yData[i];
      if (i + 1 < 4)
        std::cout << ", ";
    }
    std::cout << "]\n";

    return 0;

  } else {
    llvm::errs() << "Unknown mode '" << mode
                 << "'. Expected 'linear-int' or 'linear-mem'.\n";
    return 1;
  }
}
