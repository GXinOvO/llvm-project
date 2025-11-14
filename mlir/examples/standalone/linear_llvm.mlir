module {
  llvm.func @test(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
    %0 = llvm.mul %arg0, %arg1 : i32
    %1 = llvm.add %0, %arg2 : i32
    llvm.return %1 : i32
  }
}

