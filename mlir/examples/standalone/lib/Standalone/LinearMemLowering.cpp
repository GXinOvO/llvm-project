//===- LinearLowering.cpp - Lower standalone.linear -------------*- C++ -*-===//
//
// standalone.linear %A, %x, %y
//   (A: memref<MxKxf32>, x: memref<Kxf32>, y: memref<Mxf32>)
// => 两层 for 循环 + memref.load/store + arith.mulf/addf
//
//===----------------------------------------------------------------------===//

#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"

using namespace mlir;

namespace mlir::standalone {

namespace {

struct LinearMemOpLowering : public OpRewritePattern<LinearMemOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LinearMemOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value A = op.getMatrix();
    Value x = op.getVector();
    Value y = op.getResult();

    auto typeA = llvm::dyn_cast<MemRefType>(A.getType());
    auto typeX = llvm::dyn_cast<MemRefType>(x.getType());
    auto typeY = llvm::dyn_cast<MemRefType>(y.getType());

    if (!typeA || !typeX || !typeY) {
      return rewriter.notifyMatchFailure(
          op, "expected memref types for A, x, y");
    }
    if (typeA.getRank() != 2 || typeX.getRank() != 1 || typeY.getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected A: rank-2, x: rank-1, y: rank-1 memrefs");
    }

    int64_t M = typeA.getShape()[0]; // 行数
    int64_t K = typeA.getShape()[1]; // 列数

    if (ShapedType::isDynamic(M) || ShapedType::isDynamic(K)) {
      return rewriter.notifyMatchFailure(
          op, "only static shapes supported in this example");
    }

    auto idxType = rewriter.getIndexType();
    (void)idxType;

    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cM = rewriter.create<arith::ConstantIndexOp>(loc, M);
    Value cK = rewriter.create<arith::ConstantIndexOp>(loc, K);

    rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, c0, cM, c1, ValueRange{},
        [&](OpBuilder &b, Location forLoc, Value i, ValueRange) {
          Value zeroF = b.create<arith::ConstantOp>(
              forLoc, b.getF32Type(),
              b.getFloatAttr(b.getF32Type(), 0.0));

          auto innerFor = b.create<scf::ForOp>(
              forLoc, c0, cK, c1, ValueRange{zeroF},
              [&](OpBuilder &b2, Location innerLoc, Value j,
                  ValueRange iterArgs) {
                Value sum = iterArgs[0];

                Value a_ij = b2.create<memref::LoadOp>(innerLoc, A,
                                                       ValueRange{i, j});
                Value x_j =
                    b2.create<memref::LoadOp>(innerLoc, x, ValueRange{j});

                Value prod = b2.create<arith::MulFOp>(innerLoc, a_ij, x_j);
                Value newSum =
                    b2.create<arith::AddFOp>(innerLoc, sum, prod);

                b2.create<scf::YieldOp>(innerLoc, newSum);
              });

          Value finalSum = innerFor.getResult(0);
          b.create<memref::StoreOp>(forLoc, finalSum, y, ValueRange{i});
          b.create<scf::YieldOp>(forLoc);
        });

    return success();
  }
};

struct LowerLinearMemPass
    : public PassWrapper<LowerLinearMemPass, OperationPass<func::FuncOp>> {

  StringRef getArgument() const override {
    return "standalone-lower-linear-mem";
  }

  StringRef getDescription() const override {
    return "Lower standalone.linear_mem (A*x) to nested scf.for + arith/memref";
  }

  StringRef getName() const override {
    return "LowerLinearMemPass";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LinearMemOpLowering>(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerLinearMemPass() {
  return std::make_unique<LowerLinearMemPass>();
}

} // namespace mlir::standalone
