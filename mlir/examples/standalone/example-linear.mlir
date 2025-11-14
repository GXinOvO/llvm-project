func.func @test(%x : i32, %w : i32, %b : i32) -> i32 {
  %0 = standalone.linear %x, %w, %b : i32
  return %0 : i32
}
