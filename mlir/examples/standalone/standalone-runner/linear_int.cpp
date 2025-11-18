#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

// 注册 builtin / LLVM Dialect -> LLVM IR 的翻译
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/Support/TargetSelect.h"

#include <iostream>

using namespace mlir;

int main(int argc, char **argv) {
  // 初始化本机 JIT target
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  if (argc < 5) {
    llvm::errs() << "Usage: standalone-runner <lowered-llvm.mlir> x w b\n";
    return 1;
  }

  MLIRContext context;
  context.loadDialect<mlir::BuiltinDialect>();
  context.loadDialect<LLVM::LLVMDialect>();

  // 注册 dialect 翻译到 LLVM IR
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  // 解析 MLIR 模块
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module) {
    llvm::errs() << "ERROR: failed to parse MLIR file: " << argv[1] << "\n";
    return 1;
  }

  // 创建 ExecutionEngine（JIT）
  auto engineOrErr = ExecutionEngine::create(*module);
  if (!engineOrErr) {
    llvm::errs() << "ERROR: failed to create ExecutionEngine\n";
    return 1;
  }
  std::unique_ptr<ExecutionEngine> engine = std::move(*engineOrErr);

  // 查找函数 @test
  auto fPtrOrErr = engine->lookup("test");
  if (!fPtrOrErr) {
    llvm::errs() << "ERROR: function @test not found in module\n";
    return 1;
  }

  using FuncPtr = int (*)(int, int, int);
  FuncPtr fn = reinterpret_cast<FuncPtr>(*fPtrOrErr);

  // 从命令行读取 x, w, b
  int x = std::stoi(argv[2]);
  int w = std::stoi(argv[3]);
  int b = std::stoi(argv[4]);

  int result = fn(x, w, b);
  std::cout << "Result = " << result << "\n";

  return 0;
}
