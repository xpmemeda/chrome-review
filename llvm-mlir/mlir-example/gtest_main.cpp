#include "gtest/gtest.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Linalg/Passes.h"

#include "dialect.h"
#include "passes.h"

using namespace mlir::demo;

class MlirDemoTest : public ::testing::Test {
protected:
  MlirDemoTest() : m_context(), m_builder(&m_context) {
    llvm::DebugFlag = 1;
  }

  ~MlirDemoTest() override {}

  void SetUp() override {
    m_context.getOrLoadDialect<mlir::demo::DemoDialect>();
    m_context.getOrLoadDialect<mlir::StandardOpsDialect>();
    m_context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc());
    auto func =
        mlir::FuncOp::create(m_builder.getUnknownLoc(), "mlir_main",
                             m_builder.getFunctionType(llvm::None, llvm::None));
    if (!func) {
      llvm::errs() << "Failed to create MLIR FuncOp\n";
    }
    auto entryBlock = func.addEntryBlock();
    m_builder.setInsertionPointToStart(entryBlock);
    m_module.push_back(func);
  }

  void TearDown() override {
  }

  void SetReturnValues(mlir::Location loc, mlir::ArrayRef<mlir::Value> values) {
    auto return_op = m_builder.create<mlir::ReturnOp>(loc, values);
    auto entry_func = GetEntryFunc();
    entry_func.setType(m_builder.getFunctionType(
        entry_func.getType().getInputs(), return_op.getOperandTypes()));
  }

  mlir::FuncOp GetEntryFunc() {
    mlir::FuncOp entry_func;
    m_module->walk([&entry_func](mlir::Operation *op) {
      mlir::FuncOp func = llvm::dyn_cast<mlir::FuncOp>(op);
      if (func && func.getName() == "mlir_main") {
        entry_func = func;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    return entry_func;
  }

protected:
  mlir::MLIRContext m_context;
  mlir::OpBuilder m_builder;
  mlir::ModuleOp m_module;
};

TEST_F(MlirDemoTest, Empty) {
  // Note: just insert a terminator
  SetReturnValues(m_builder.getUnknownLoc(), {});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "[demo dialect -> llvm dialect]\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));
}

TEST_F(MlirDemoTest, ReturnConstant) {
  // Note: TensorType is not compatiable!
  auto dtype = mlir::VectorType::get({3, 3}, m_builder.getF64Type());
  auto data = mlir::DenseElementsAttr::get(dtype, 0.1);
  mlir::Value c = m_builder.create<mlir::ConstantOp>(m_builder.getUnknownLoc(), data);
  SetReturnValues(m_builder.getUnknownLoc(), {c});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "[demo dialect -> llvm dialect]\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "[mlir -> llvm ir]\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";
}

TEST_F(MlirDemoTest, ReturnArgs1) {
  llvm::outs() << "==================== demo dialect ================\n";
  GetEntryFunc().insertArgument(0, mlir::VectorType::get({3, 3}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  SetReturnValues(m_builder.getUnknownLoc(), GetEntryFunc().getArgument(0));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "=============== llvm dialect =====================\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================= llvm ir ==========================\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";
}

TEST_F(MlirDemoTest, ReturnArgs2) {
  GetEntryFunc().insertArgument(0, mlir::MemRefType::get({3, 3}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  SetReturnValues(m_builder.getUnknownLoc(), GetEntryFunc().getArgument(0));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "[demo dialect -> llvm dialect]\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "[mlir -> llvm ir]\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";
}

TEST_F(MlirDemoTest, Elemwise) {
  mlir::Value a = m_builder.create<mlir::arith::ConstantFloatOp>(
      m_builder.getUnknownLoc(), llvm::APFloat{0.1}, m_builder.getF64Type());
  mlir::Value b = m_builder.create<mlir::arith::ConstantFloatOp>(
      m_builder.getUnknownLoc(), llvm::APFloat{1.0}, m_builder.getF64Type());
  mlir::Value c = m_builder.create<mlir::arith::ConstantFloatOp>(
      m_builder.getUnknownLoc(), llvm::APFloat{10.}, m_builder.getF64Type());
  mlir::Value d =
      m_builder.create<ScalarF64AddOp>(m_builder.getUnknownLoc(), a, b);
  mlir::Value e =
      m_builder.create<ScalarF64MulOp>(m_builder.getUnknownLoc(), c, d);
  m_builder.create<PrintOp>(m_builder.getUnknownLoc(), e);
  SetReturnValues(m_builder.getUnknownLoc(), {e});
  m_module.dump();

  // lower to llvm dialect
  llvm::outs() << "demo dialect -> llvm dialect\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();

  // mlir -> llvm ir
  llvm::outs() << "mlir -> llvm ir\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";

  llvm::outs() << "run llvm ir\n";
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(m_context);

  // An optimization pipeline to use within the execution engine.
  bool enableOpt = false;
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      m_module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();
  auto invocationResult = engine->invokePacked("mlir_main");
  ASSERT_TRUE(bool(invocationResult));
}

TEST_F(MlirDemoTest, PointWiseNegOp) {
  llvm::outs() << "=================== demo dialect ===============\n";
  // Note: MemRefType is not compatalbe.
  GetEntryFunc().insertArgument(0, mlir::VectorType::get({3, 3}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  mlir::Value a = m_builder.create<PointWiseNegOp>(m_builder.getUnknownLoc(), GetEntryFunc().getArgument(0));
  SetReturnValues(m_builder.getUnknownLoc(), {a});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "============== llvm dialect ===================\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "=================== llvm ir =====================\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";
}

TEST_F(MlirDemoTest, PointWiseExp) {
  llvm::outs() << "=================== demo dialect ===============\n";
  auto dataType = mlir::VectorType::get({3, 3}, m_builder.getF32Type());
  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, 0.1f);
  mlir::Value c = m_builder.create<mlir::ConstantOp>(
    m_builder.getUnknownLoc(), dataAttribute);
  mlir::Value a = m_builder.create<PointWiseExpOp>(m_builder.getUnknownLoc(), c);
  SetReturnValues(m_builder.getUnknownLoc(), {a});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "============== llvm dialect ===================\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "=================== llvm ir =====================\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";
}

TEST_F(MlirDemoTest, TransposeOp) {
  llvm::outs() << "=============== demo dialect ===================\n";
  GetEntryFunc().insertArgument(0, mlir::VectorType::get({3, 3}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  mlir::Value a = m_builder.create<TransposeOp>(m_builder.getUnknownLoc(), GetEntryFunc().getArgument(0));
  SetReturnValues(m_builder.getUnknownLoc(), {a});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== vector dialect ===================\n";
  mlir::PassManager vpm(&m_context);
  vpm.addPass(mlir::demo::createLowerToVectorPass());
  ASSERT_FALSE(mlir::failed(vpm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== llvm dialect ===================\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== llvm ir ========================\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";
}

TEST_F(MlirDemoTest, PrintOp) {
  llvm::outs() << "=============== demo dialect ===================\n";
  auto dtype = mlir::VectorType::get({9}, m_builder.getF64Type());
  auto data = mlir::DenseElementsAttr::get(dtype, 0.1);
  mlir::Value c = m_builder.create<mlir::ConstantOp>(m_builder.getUnknownLoc(), data);
  m_builder.create<PrintVecOp>(m_builder.getUnknownLoc(), c);
  SetReturnValues(m_builder.getUnknownLoc(), {});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== vector dialect ===================\n";
  mlir::PassManager vpm(&m_context);
  vpm.addPass(mlir::demo::createLowerToVectorPass());
  ASSERT_FALSE(mlir::failed(vpm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== llvm dialect ===================\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== llvm ir ========================\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";

  llvm::outs() << "==================== run llvm ir ==========================\n";
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(m_context);

  // An optimization pipeline to use within the execution engine.
  bool enableOpt = false;
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      m_module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();
  auto invocationResult = engine->invokePacked("mlir_main");
  ASSERT_TRUE(bool(invocationResult));
}

TEST_F(MlirDemoTest, MatmulOp2) {
  llvm::outs() << "=============== demo dialect ===================\n";
  GetEntryFunc().insertArgument(0, mlir::MemRefType::get({3, 2}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  GetEntryFunc().insertArgument(1, mlir::MemRefType::get({2, 4}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  mlir::Value matmul = m_builder.create<MatmulOp>(m_builder.getUnknownLoc(), GetEntryFunc().getArgument(0), GetEntryFunc().getArgument(1));
  SetReturnValues(m_builder.getUnknownLoc(), {matmul});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== linalg dialect ===================\n";
  mlir::PassManager pm1(&m_context);
  pm1.addPass(mlir::demo::createLowerToLinalgPass());
  ASSERT_TRUE(mlir::succeeded(pm1.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== affine dialect ====================\n";
  mlir::PassManager pm2(&m_context);
  pm2.addNestedPass<mlir::FuncOp>(mlir::createLinalgBufferizePass());
  pm2.addNestedPass<mlir::FuncOp>(mlir::createConvertLinalgToAffineLoopsPass());
  ASSERT_TRUE(mlir::succeeded(pm2.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== llvm dialect ======================\n";
  mlir::PassManager pm3(&m_context);
  pm3.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm3.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));
}

TEST_F(MlirDemoTest, MatmulOp) {
  llvm::outs() << "=============== demo dialect ===================\n";
  auto dtype = mlir::VectorType::get({9}, m_builder.getF64Type());
  auto data = mlir::DenseElementsAttr::get(dtype, 0.1);
  mlir::Value c = m_builder.create<mlir::ConstantOp>(m_builder.getUnknownLoc(), data);
  mlir::Value a = m_builder.create<MatmulOp>(m_builder.getUnknownLoc(), c, c);
  SetReturnValues(m_builder.getUnknownLoc(), {a});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== vector dialect ===================\n";
  mlir::PassManager vpm(&m_context);
  vpm.addPass(mlir::demo::createLowerToVectorPass());
  ASSERT_FALSE(mlir::failed(vpm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== llvm dialect ===================\n";
  mlir::PassManager pm(&m_context);
  pm.addPass(mlir::demo::createLowerToLlvmPass());
  ASSERT_FALSE(mlir::failed(pm.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== llvm ir ========================\n";
  mlir::registerLLVMDialectTranslation(m_context);
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(m_module, llvmContext);
  ASSERT_TRUE(llvmModule);
  llvm::errs() << *llvmModule << "\n";

  llvm::outs() << "==================== run llvm ir ==========================\n";
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(m_context);

  // An optimization pipeline to use within the execution engine.
  bool enableOpt = false;
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      m_module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();
  auto invocationResult = engine->invokePacked("mlir_main");
  ASSERT_TRUE(bool(invocationResult));
}

TEST_F(MlirDemoTest, DoubleNeg) {
  llvm::outs() << "=============== demo dialect ===================\n";
  GetEntryFunc().insertArgument(0, mlir::MemRefType::get({3, 2}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  mlir::Value neg1 = m_builder.create<PointWiseNegOp>(m_builder.getUnknownLoc(), GetEntryFunc().getArgument(0));
  mlir::Value neg2 = m_builder.create<PointWiseNegOp>(m_builder.getUnknownLoc(), neg1);
  SetReturnValues(m_builder.getUnknownLoc(), {neg2});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== linalg dialect ===================\n";
  mlir::PassManager pm1(&m_context);
  pm1.addPass(mlir::demo::createLowerToLinalgPass());
  ASSERT_TRUE(mlir::succeeded(pm1.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

}

TEST_F(MlirDemoTest, DoubleNeg2) {
  llvm::outs() << "=============== demo dialect ===================\n";
  GetEntryFunc().insertArgument(0, mlir::MemRefType::get({3, 2}, m_builder.getF32Type()), {}, m_builder.getUnknownLoc());
  mlir::Value neg1 = m_builder.create<PointWiseNegOp>(m_builder.getUnknownLoc(), GetEntryFunc().getArgument(0));
  mlir::Value neg2 = m_builder.create<PointWiseExpOp>(m_builder.getUnknownLoc(), neg1);
  SetReturnValues(m_builder.getUnknownLoc(), {neg2});
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

  llvm::outs() << "================== linalg dialect ===================\n";
  mlir::PassManager pm1(&m_context);
  pm1.addPass(mlir::demo::createLowerToLinalgPass());
  ASSERT_TRUE(mlir::succeeded(pm1.run(m_module)));
  m_module.dump();
  ASSERT_TRUE(mlir::succeeded(mlir::verify(m_module)));

}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}