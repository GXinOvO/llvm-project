module {
  func.func @test(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %0 = arith.muli %arg0, %arg1 : i32
    %1 = arith.addi %0, %arg2 : i32
    return %1 : i32
  }
}

