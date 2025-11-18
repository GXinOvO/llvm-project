set(LIBS
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRStandalone
)

add_llvm_executable(linear-int-runner linear_int.cpp)

llvm_update_compile_flags(linear-int-runner)
target_link_libraries(linear-int-runner  PRIVATE ${LIBS})

mlir_check_all_link_libraries(linear-int-runner )
