- 创建函数

```cpp
auto func = mlir::FuncOp::create(builder.getUnknownLoc(), "mlir_main", builder.getFunctionType(llvm::None, llvm::None));
```

- 创建常量

```cpp
mlir::Value a = builder.create<mlir::ConstantFloatOp>(builder.getUnknownLoc(), llvm::APFloat{0.1}, builder.getF64Type());
```

- 指定函数返回值

```cpp
builder.create<mlir::ReturnOp>(builder.getUnknownLoc(), llvm::ArrayRef<mlir::Value>{});
```
