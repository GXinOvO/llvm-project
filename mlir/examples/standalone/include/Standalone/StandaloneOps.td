//===- StandaloneOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "Standalone/StandaloneTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Standalone_FooOp : Standalone_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

/*
  TableGen是LLVM项目中用于代码生成(code generation)的一种领域特定语言(DSL)，它本身不是通用编程语言，而是一种声明式语言，
专门用于描述指令集、寄存器、操作码、调度信息等底层编译器相关的结构。

TableGen的主要用途:
  1. 自动生成C++代码: LLVM使用TableGen来避免手动编写大量重复、易错的C++代码
  2. 定义目标架构信息: 例如在为新CPU架构(如RISC-V、ARM、x86)编写后端时，用.td(TableGen Description)文件描述该架构的指令
格式、寄存器文件、指令编码规则等
  3. 支持编译器组件生成: 包括指令选择器、调度模型、汇编打印/解析器、MC层信息等
*/

/*
  · def: 定义一个东西，名字叫Standalone_LinearOp
  · 右边Standalone_Op<"Linear", [Pure]>:
    · 表示它属于Standalone dialect，操作名叫"linear"
    · [Pure] 表示这个op没有副作用(不修改全局状态，不I/O)，类似数学函数

  会自动识别summary和description
*/
def Standalone_LinearOp : Standalone_Op<"linear", [Pure]> {
  // 一行简介，后面mlir-opt --help时会用到
  let summary = "A simple scalar linear op: y = x * w + b";
  // 一大段文档(支持多行)，也是给help/文档看的，不影响实际语义，只是说明
  let description = [{
    standalone.linear %x, %w, %b : i32

    is lowered to:

      %m = arith.muli %x, %w : i32
      %y = arith.addi %m, %b : i32
  }];

  // 三个 i32 输入，一个 i32 输出


  /*
    这一段直接决定了:
      · 生成C++代码时，会有Value getInput()，getWeight()，getBias()等函数
      · MLIR里这个op必须用三个i32当输入，一个i32当输出  
  */
  // 这里有三个入参，都叫I32(32位整数)，名字分别是$input/$weight/$bias
  let arguments = (ins I32:$input,
                       I32:$weight,
                       I32:$bias);
  // 这里只有一个I32的输出，名为$output
  let results   = (outs I32:$output);

  // 自定义打印格式：
  //   standalone.linear %x, %w, %b : i32
  // 等同于 %y = standalone.linear %x, %w, %b : i32
  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict `:` type($output)
  }];
}


#endif // STANDALONE_OPS
