// RUN: standalone-opt %s --standalone-lower-linear-mem 
// 动态形状版本：
//   A : memref<?x?xf32>   // M x K
//   x : memref<?xf32>     // K
//   y : memref<?xf32>     // M
//
// 具体的 M、K 在运行时由 C++ 传入。

func.func @main(%A : memref<?x?xf32>,
                %x : memref<?xf32>,
                %y : memref<?xf32>)
    attributes { llvm.emit_c_interface } {

  // y = A * x  （具体怎么算交给 standalone.linear_mem 的 runtime 实现）
  standalone.linear_mem %A, %x, %y
      : memref<?x?xf32>, memref<?xf32>, memref<?xf32>

  return
}