#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>
#include <vector>

using namespace mlir;

// =========================
// 小工具：从文件加载 MLIR Module
// =========================
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

// =========================
// 表示一块 memref<...xf32> 的实际数据
// =========================
struct MemRefBuffer {
  std::vector<float> data;               // 连续存放所有元素
  SmallVector<int64_t, 4> shape;         // 维度信息，例如 [M, K]
};

// 运行时环境：把 IR 里的 Value 绑定到 MemRefBuffer*
using ValueEnv = llvm::DenseMap<Value, MemRefBuffer *>;

// =========================
// 一个简单的矩阵 × 向量 C 内核：y = A * x
// A: [M x K], x: [K], y: [M]
// =========================
static void runLinearMemKernel(const float *A,
                               const float *x,
                               float *y,
                               int64_t M,
                               int64_t K) {
  for (int64_t i = 0; i < M; ++i) {
    float sum = 0.0f;
    for (int64_t j = 0; j < K; ++j) {
      sum += A[i * K + j] * x[j];
    }
    y[i] = sum;
  }
}

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);

  if (argc < 3) {
    llvm::errs()
        << "Usage:\n"
        << "  " << argv[0]
        << " linear-int <mlir-file> x w b\n"
        << "  " << argv[0]
        << " linear-mem <mlir-file>\n"
        << "  " << argv[0]
        << " linear-mem-dyn <mlir-file> M K\n";
    return 1;
  }

  std::string mode = argv[1];
  llvm::StringRef inputFilename = argv[2];

  // =========================
  // 1. 建立 MLIR Context，并注册会用到的 dialect
  // =========================
  DialectRegistry registry;
  registry.insert<func::FuncDialect,
                  memref::MemRefDialect,
                  standalone::StandaloneDialect>();
  MLIRContext context(registry);

  // 解析 MLIR 文件为 ModuleOp
  OwningOpRef<ModuleOp> module = loadModule(inputFilename, context);
  if (!module)
    return 1;

  // 一个池子，保存所有全局 MemRefBuffer 的所有权，保证生命周期
  std::vector<std::unique_ptr<MemRefBuffer>> globalBuffers;
  // 名字 -> MemRefBuffer*（用于 memref.get_global）
  llvm::StringMap<MemRefBuffer *> globalEnv;

  // =========================
  // 2. 扫描 memref.global，把 dense<...> 的 float 常量读出来
  //    注意：linear-mem-dyn 模式可以不依赖 global，这段逻辑
  //    对于 example-linear_mem-dyn.mlir（没有 memref.global）会直接跳过
  // =========================
  for (auto globalOp : module->getOps<memref::GlobalOp>()) {
    Type globalType = globalOp.getType();
    auto type = llvm::dyn_cast<MemRefType>(globalType);
    if (!type)
      continue;

    Type elemTy = type.getElementType();
    if (!llvm::isa<FloatType>(elemTy))
      continue;

    auto initAttrOpt = globalOp.getInitialValue();
    if (!initAttrOpt.has_value())
      continue;
    Attribute initAttr = *initAttrOpt;

    auto denseAttr = llvm::dyn_cast<DenseFPElementsAttr>(initAttr);
    if (!denseAttr)
      continue;

    auto buf = std::make_unique<MemRefBuffer>();
    buf->shape = SmallVector<int64_t, 4>(type.getShape().begin(),
                                         type.getShape().end());

    buf->data.reserve(denseAttr.size());
    for (auto it : denseAttr) {
      buf->data.push_back(it.convertToFloat());
    }

    MemRefBuffer *rawPtr = buf.get();
    globalEnv[globalOp.getSymName()] = rawPtr;
    globalBuffers.push_back(std::move(buf));

    llvm::outs() << "Loaded global @" << globalOp.getSymName()
                 << " with " << denseAttr.size() << " elements\n";
  }

  // =========================
  // 3. 分三种模式处理
  // =========================
  if (mode == "linear-int") {
    // =========================
    // 模式一：linear-int
    // =========================
    if (argc < 6) {
      llvm::errs()
          << "Usage: " << argv[0]
          << " linear-int <mlir-file> x w b\n";
      return 1;
    }

    int x = std::stoi(argv[3]);
    int w = std::stoi(argv[4]);
    int b = std::stoi(argv[5]);

    int result = x * w + b;
    std::cout << "Result = " << result
              << "  (computed by C++ directly)\n";
    return 0;

  } else if (mode == "linear-mem") {
    // =========================
    // 模式二：linear-mem （静态 global + 解释器）
    // =========================

    ValueEnv valueEnv;

    func::FuncOp mainFunc =
        module->lookupSymbol<func::FuncOp>("main");
    if (!mainFunc) {
      llvm::errs() << "Error: cannot find func @main in module\n";
      return 1;
    }

    Block &entryBlock = mainFunc.getBody().front();

    for (Operation &op : entryBlock) {
      if (auto getg = llvm::dyn_cast<memref::GetGlobalOp>(&op)) {
        StringRef name = getg.getName();
        auto it = globalEnv.find(name);
        if (it == globalEnv.end()) {
          llvm::errs() << "Global " << name
                       << " not found in globalEnv\n";
          return 1;
        }

        Value result = getg.getResult();
        valueEnv[result] = it->second;

        llvm::outs() << "Bind Value from memref.get_global @" << name
                     << " to MemRefBuffer\n";
        continue;
      }

      if (auto linear = llvm::dyn_cast<standalone::LinearMemOp>(&op)) {
        auto aType = llvm::dyn_cast<MemRefType>(linear.getMatrix().getType());
        if (!aType || aType.getRank() != 2) {
          llvm::errs() << "A is not memref<2D>\n";
          return 1;
        }
        int64_t M = aType.getShape()[0];
        int64_t K = aType.getShape()[1];

        llvm::outs() << "standalone.linear_mem: detected shape M=" << M
                     << ", K=" << K << "\n";

        Value aVal = linear.getMatrix();
        auto itBuf = valueEnv.find(aVal);
        if (itBuf == valueEnv.end()) {
          llvm::errs() << "No buffer bound for A value\n";
          return 1;
        }
        MemRefBuffer *A_buf = itBuf->second;
        if (A_buf->data.size() != static_cast<size_t>(M * K)) {
          llvm::errs() << "A buffer size mismatch: expect "
                       << (M * K) << ", got "
                       << A_buf->data.size() << "\n";
          return 1;
        }

        std::vector<float> xData(K, 1.0f);
        std::vector<float> yData(M, 0.0f);

        runLinearMemKernel(A_buf->data.data(),
                           xData.data(),
                           yData.data(),
                           M, K);

        std::cout << "y = [";
        for (int64_t i = 0; i < M; ++i) {
          std::cout << yData[i];
          if (i + 1 < M)
            std::cout << ", ";
        }
        std::cout << "]  (A from memref.global, x from C++)\n";

        llvm::outs() << "\n=== MLIR dump of y_result ===\n";
        llvm::outs() << "module {\n";
        llvm::outs() << "  memref.global \"private\" @y_result : memref<"
                     << M << "xf32> = dense<[";
        for (int64_t i = 0; i < M; ++i) {
          llvm::outs() << yData[i];
          if (i + 1 < M)
            llvm::outs() << ", ";
        }
        llvm::outs() << "]>\n";
        llvm::outs() << "}\n";

        return 0;
      }
    }

    llvm::errs() << "No standalone.linear_mem op found in @main\n";
    return 1;

  } else if (mode == "linear-mem-dyn") {
    // =========================
    // 模式三：linear-mem-dyn （动态形状）
    //
    // main 的签名：
    //   func.func @main(%A: memref<?x?xf32>,
    //                   %x: memref<?xf32>,
    //                   %y: memref<?xf32>) { ... }
    //
    //   A : M x K
    //   x : K
    //   y : M
    //
    // 具体 M/K 由命令行参数提供。
    // =========================

    if (argc < 5) {
      llvm::errs()
          << "Usage: " << argv[0]
          << " linear-mem-dyn <mlir-file> M K\n";
      return 1;
    }

    int64_t M = std::stoll(argv[3]);  // 行数
    int64_t K = std::stoll(argv[4]);  // 列数

    if (M <= 0 || K <= 0) {
      llvm::errs() << "M and K must be positive\n";
      return 1;
    }

    func::FuncOp mainFunc =
        module->lookupSymbol<func::FuncOp>("main");
    if (!mainFunc) {
      llvm::errs() << "Error: cannot find func @main in module\n";
      return 1;
    }

    Block &entryBlock = mainFunc.getBody().front();

    if (entryBlock.getNumArguments() != 3) {
      llvm::errs()
          << "Expect @main to have 3 arguments (A,x,y), got "
          << entryBlock.getNumArguments() << "\n";
      return 1;
    }

    Value aVal = entryBlock.getArgument(0);
    Value xVal = entryBlock.getArgument(1);
    Value yVal = entryBlock.getArgument(2);

    auto aType = llvm::dyn_cast<MemRefType>(aVal.getType());
    auto xType = llvm::dyn_cast<MemRefType>(xVal.getType());
    auto yType = llvm::dyn_cast<MemRefType>(yVal.getType());
    if (!aType || !xType || !yType) {
      llvm::errs() << "Arguments of @main must be memref types\n";
      return 1;
    }

    // 这里不强行检查 shape，因为它们是 ?，
    // 只要求 rank 正确即可：
    if (aType.getRank() != 2 || xType.getRank() != 1 || yType.getRank() != 1) {
      llvm::errs() << "@main arg types must be "
                   << "memref<?x?xf32>, memref<?xf32>, memref<?xf32>\n";
      return 1;
    }

    // ---- 动态分配 A / x / y buffer，并初始化 ----
    auto bufA = std::make_unique<MemRefBuffer>();
    bufA->shape = {M, K};
    bufA->data.resize(M * K);

    int idx = 0;
    for (int64_t i = 0; i < M; ++i) {
      for (int64_t j = 0; j < K; ++j) {
        bufA->data[i * K + j] = static_cast<float>(++idx);  // 1..M*K
      }
    }

    auto bufX = std::make_unique<MemRefBuffer>();
    bufX->shape = {K};
    bufX->data.resize(K, 1.0f);  // x = [1,1,...]

    auto bufY = std::make_unique<MemRefBuffer>();
    bufY->shape = {M};
    bufY->data.resize(M, 0.0f);  // y = [0,0,...]

    MemRefBuffer *A_buf = bufA.get();
    MemRefBuffer *X_buf = bufX.get();
    MemRefBuffer *Y_buf = bufY.get();

    // 为了让它们生命周期够长，用一个 vector 持有
    std::vector<std::unique_ptr<MemRefBuffer>> localBuffers;
    localBuffers.push_back(std::move(bufA));
    localBuffers.push_back(std::move(bufX));
    localBuffers.push_back(std::move(bufY));

    // ---- 构造 ValueEnv，把 block argument 绑定到 buffer 上 ----
    ValueEnv valueEnv;
    valueEnv[aVal] = A_buf;
    valueEnv[xVal] = X_buf;
    valueEnv[yVal] = Y_buf;

    // ---- 解释执行 main 里的 op ----
    for (Operation &op : entryBlock) {
      if (auto linear = llvm::dyn_cast<standalone::LinearMemOp>(&op)) {
        llvm::outs()
            << "standalone.linear_mem (dynamic): M=" << M
            << ", K=" << K << "\n";

        // 从 valueEnv 取出对应 buffer
        auto itA = valueEnv.find(linear.getMatrix());
        auto itX = valueEnv.find(linear.getVector());
        auto itY = valueEnv.find(linear.getResult());

        if (itA == valueEnv.end() || itX == valueEnv.end() ||
            itY == valueEnv.end()) {
          llvm::errs() << "Some operands of standalone.linear_mem "
                          "do not have bound buffers\n";
          return 1;
        }

        A_buf = itA->second;
        X_buf = itX->second;
        Y_buf = itY->second;

        if (A_buf->data.size() != static_cast<size_t>(M * K)) {
          llvm::errs() << "A buffer size mismatch in dynamic mode\n";
          return 1;
        }
        if (X_buf->data.size() != static_cast<size_t>(K)) {
          llvm::errs() << "x buffer size mismatch in dynamic mode\n";
          return 1;
        }
        if (Y_buf->data.size() != static_cast<size_t>(M)) {
          llvm::errs() << "y buffer size mismatch in dynamic mode\n";
          return 1;
        }

        // 运行 C kernel
        runLinearMemKernel(A_buf->data.data(),
                           X_buf->data.data(),
                           Y_buf->data.data(),
                           M, K);

        // 打印 y 数值
        std::cout << "y = [";
        for (int64_t i = 0; i < M; ++i) {
          std::cout << Y_buf->data[i];
          if (i + 1 < M)
            std::cout << ", ";
        }
        std::cout << "]  (dynamic shape, A/x/y from runtime)\n";

        // Dump 成一个合法 MLIR module
        llvm::outs() << "\n=== MLIR dump of y_result (dynamic) ===\n";
        llvm::outs() << "module {\n";
        llvm::outs() << "  memref.global \"private\" @y_result : memref<"
                     << M << "xf32> = dense<[";
        for (int64_t i = 0; i < M; ++i) {
          llvm::outs() << Y_buf->data[i];
          if (i + 1 < M)
            llvm::outs() << ", ";
        }
        llvm::outs() << "]>\n";
        llvm::outs() << "}\n";

        return 0;
      }
      // 其他 op（比如 func.return）这里暂时忽略
    }

    llvm::errs() << "No standalone.linear_mem op found in @main "
                 << "for linear-mem-dyn mode\n";
    return 1;

  } else {
    llvm::errs() << "Unknown mode '" << mode
                 << "'. Expected 'linear-int', 'linear-mem' or 'linear-mem-dyn'.\n";
    return 1;
  }
}
