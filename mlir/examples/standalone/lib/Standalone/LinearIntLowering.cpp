//===- LinearLowering.cpp - Lower standalone.linear ---------*- C++ -*-===//
//
//  把 `standalone.linear` 降成：
//    %m = arith.muli %x, %w : i32
//    %y = arith.addi %m, %b : i32
//
//===----------------------------------------------------------------------===//

// 引入我的dialect和算子声明(LinearOp的类定义在这里)
#include "Standalone/StandaloneDialect.h"
#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

// 等下要创建arith.muli和arith.addi和操作func.func
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 提供OpRewritePattern，写 匹配 + rewrite 的模版
#include "mlir/IR/PatternMatch.h"
// 定义Pass的基类
#include "mlir/Pass/Pass.h"
// 提供applyPatternsAndFoldGreadily，帮你在整个函数上反复应用pattern
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// Pass + Pattern 告诉编译器 standalone.linear = 乘 + 加

using namespace mlir;

namespace mlir::standalone {

namespace {
/*
  LinearOpLowering: 定义的改写规则类
  继承自OpRewritePattern<LinearOp>，意味着我专门负责把LinearOp改写
 */
struct LinearIntOpLowering : public OpRewritePattern<LinearIntOp> {

  // 把父类构造函数接过来用，方便外面初始化这个pattern
  using OpRewritePattern::OpRewritePattern;

  // matchAndRewrite: 核心逻辑，遇到一个LinearOp就会调用一次这里
  // op: 匹配到的那条standalone.linear;   rewrite: 帮你插入到新op、替换旧op的工具
  LogicalResult matchAndRewrite(LinearIntOp op,
                                PatternRewriter &rewriter) const override {
    // getLoc(): 源代码位置，用于报错/调试，可继续沿用
    Location loc = op.getLoc();
    // 都是TableGen自动生成的getter
    // 这些Value就是MLIR里的SSA值
    Value x = op.getInput();
    Value w = op.getWeight();
    Value b = op.getBias();

    // m = x * w
    // 插入一条新指令: $0 = arith.muli %x, %w : i32
    auto mul = rewriter.create<arith::MulIOp>(loc, x, w);
    // y = m + b
    auto add = rewriter.create<arith::AddIOp>(loc, mul, b);

    // 意思是: 以后凡是用到op输出的地方，改成用add的结果。然后把原来的standalone.linear删除
    rewriter.replaceOp(op, add.getResult());
    // 告诉pattern引擎: 这次改写成功了
    return success();
  }
};

// 把pattern应用到整个函数上
// 这是我们真正挂到管线里的Pass
/*
  PassWrapper是MLIR提供的模板类，用于简化自定义pass的编写
  第一个模板参数LowerLinearPass: CRTP(奇异递归模板模式)，让基类能调用派生类方法
  第二类参数OperationPass<func::FuncOp>: 指定这个pass作用于func.func操作
    即: 每次调用runOnOperation时, 当前操作就是func.func
 */
struct LowerLinearIntPass
    : public PassWrapper<LowerLinearIntPass, OperationPass<func::FuncOp>> {

  // ★★★ 关键：告诉 PassManager 这个 pass 的命令行名字 ★★★
  // mlir-opt --standalone-lower-linear input.mlir就会触发这个pass
  StringRef getArgument() const override {
    // 这里必须和你在 .td 里 Pass<"..."> 的字符串一致
    return "standalone-lower-linear-int";
  }

  StringRef getDescription() const override {
    return "Lower standalone.linear_int to arith.muli + arith.addi";
  }

  StringRef getName() const override {
    return "LowerLinearIntPass";
  }

  /*
    作用: 告诉MLIR这个pass可能会生成哪些dialect的操作
    为什么需要: 
      · MLIR要求: 不能在未注册的dialect上创建操作
      · 此pass会生成arith.muli和arith.addi, 它们属于arith dialect
      · 如果不声明，运行时会报错：dialect 'arith' is not registered
  */
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LinearIntOpLowering>(ctx);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// 工厂函数，供 TableGen 里 constructor 使用
std::unique_ptr<mlir::Pass> createLowerLinearIntPass() {
  return std::make_unique<LowerLinearIntPass>();
}

} // namespace mlir::standalone
