//===- StandalonePasses.td - Standalone dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_PASS
#define STANDALONE_PASS

include "mlir/Pass/PassBase.td"

def StandaloneSwitchBarFoo: Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def LowerLinearIntPass : Pass<"standalone-lower-linear-int", "func::FuncOp"> {
  let summary = "Lower standalone.linear_int to arith.muli + arith.addi";
  let description = [{
    This pass lowers `standalone.linear_int` operations:

      y = standalone.linear_int(x, w, b)

    into:

      m = arith.muli x, w
      y = arith.addi m, b
  }];

  // 对应 C++ 工厂函数（刚刚在 LinearLowering.cpp 里实现的）
  let constructor = "mlir::standalone::createLowerLinearIntPass()";
}

def LowerLinearMemPass : Pass<"standalone-lower-linear-mem", "func::FuncOp"> {
  let summary = "Lower standalone.linear_mem (A*x) to nested loops";

  let description = [{
    二位矩阵和一维向量的乘法

    - A : memref<MxKxf32>
    - x : memref<Kxf32>
    - y : memref<Mxf32>


    语义: 对每一行 i.
      y[i] = sum_j A[i, j] * x[j]
  }];

  let constructor = "mlir::standalone::createLowerLinearMemPass()";
}

#endif // STANDALONE_PASS
