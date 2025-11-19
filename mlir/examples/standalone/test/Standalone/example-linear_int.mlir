// RUN: standalone-opt %s --standalone-lower-linear-int 
func.func @test(%x : i32, %w : i32, %b : i32) -> i32 
  attributes { llvm.emit_c_interface } {
  %0 = standalone.linear_int %x, %w, %b : i32
  return %0 : i32
}
