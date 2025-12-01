// RUN: standalone-opt %s --standalone-lower-linear-mem 
// 矩阵 A: 4x3，行优先
memref.global "private" @A_init : memref<4x3xf32> = dense<
  [[1.0,  2.0,  3.0],
   [4.0,  5.0,  6.0],
   [7.0,  8.0,  9.0],
   [10.0, 11.0, 12.0]]>

// 向量 x: 3
memref.global "private" @x_init : memref<3xf32> = dense<[1.0, 1.0, 1.0]>

// 输出向量 y: 4，先全部 0
memref.global "private" @y_init : memref<4xf32> = dense<[0.0, 0.0, 0.0, 0.0]>

func.func @main() attributes { llvm.emit_c_interface } {
  %A = memref.get_global @A_init : memref<4x3xf32>
  %x = memref.get_global @x_init : memref<3xf32>
  %y = memref.get_global @y_init : memref<4xf32>

  // 自定义算子：y = A * x
  standalone.linear_mem %A, %x, %y
      : memref<4x3xf32>, memref<3xf32>, memref<4xf32>

  return
}