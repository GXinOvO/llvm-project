set(LIBS
  MLIRExecutionEngine
  MLIRTargetLLVMIRExport
  MLIRLLVMDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRStandalone
)

add_llvm_executable(standalone-runner standalone-runner.cpp)

llvm_update_compile_flags(standalone-runner)
target_link_libraries(standalone-runner PRIVATE ${LIBS})

mlir_check_all_link_libraries(standalone-runner)
