//===- LinearLowering.cpp - Lower standalone.linear ---------*- C++ -*-===//
//
//  把 `standalone.linear` 降成：
//    %m = arith.muli %x, %w : i32
//    %y = arith.addi %m, %b : i32
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::standalone {

namespace {

struct LinearOpLowering : public OpRewritePattern<LinearOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value x = op.getInput();
    Value w = op.getWeight();
    Value b = op.getBias();

    // m = x * w
    auto mul = rewriter.create<arith::MulIOp>(loc, x, w);
    // y = m + b
    auto add = rewriter.create<arith::AddIOp>(loc, mul, b);

    rewriter.replaceOp(op, add.getResult());
    return success();
  }
};

struct LowerLinearPass
    : public PassWrapper<LowerLinearPass, OperationPass<func::FuncOp>> {

  // ★★★ 关键：告诉 PassManager 这个 pass 的命令行名字 ★★★
  StringRef getArgument() const override {
    // 这里必须和你在 .td 里 Pass<"..."> 的字符串一致
    return "standalone-lower-linear";
  }

  StringRef getDescription() const override {
    return "Lower standalone.linear to arith.muli + arith.addi";
  }

  StringRef getName() const override {
    return "LowerLinearPass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LinearOpLowering>(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// 工厂函数，供 TableGen 里 constructor 使用
std::unique_ptr<mlir::Pass> createLowerLinearPass() {
  return std::make_unique<LowerLinearPass>();
}

} // namespace mlir::standalone
