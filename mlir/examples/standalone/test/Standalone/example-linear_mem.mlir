// RUN: standalone-opt %s --standalone-lower-linear-mem 
func.func @main(%A: memref<4x3xf32>, %x: memref<3xf32>, %y: memref<4xf32>) 
    attributes { llvm.emit_c_interface } {
        standalone.linear_mem %A, %x, %y
            : memref<4x3xf32>, memref<3xf32>, memref<4xf32>
        return
}