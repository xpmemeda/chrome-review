include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Demo_Dialect : Dialect {
    let name = "demo";
    let cppNamespace = "::mlir::demo";
}

class Demo_Op<string mnemonic, list<Trait> traits = []> :
    Op<Demo_Dialect, mnemonic, traits>;

class ScalarF64Op<string mnemonic> : Demo_Op<mnemonic> {
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            $_state.addTypes(lhs.getType());
            $_state.addOperands({lhs, rhs});
        }]>
    ];
}

def ScalarF64AddOp : ScalarF64Op<"scalar_f64_add">;

def ScalarF64MulOp : ScalarF64Op<"scalar_f64_mul"> {
    // print format
    let assemblyFormat = [{
        `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)` attr-dict `to` type(results)
    }];
}

class UnaryPointWiseOp<string mnemonic> : Demo_Op<mnemonic> {
    let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>:$input);
    let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>);
    let builders = [
        OpBuilder<(ins "Value":$input), [{
            $_state.addTypes(input.getType());
            $_state.addOperands({input});
        }]>
    ];
}

def PointWiseExpOp : UnaryPointWiseOp<"pointwise_exp"> {
    let summary = "point-wise exp";
}

def PointWiseNegOp : UnaryPointWiseOp<"pointwise_neg"> {
    let summary = "point-wise neg";
}

def TransposeOp : Demo_Op<"transpose"> {
  let summary = "matrix transpose";
  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>:$input);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>);
  let builders = [
      OpBuilder<(ins "::mlir::Value":$input), [{
          $_state.addTypes(input.getType());
          $_state.addOperands({input});
      }]>
  ];
}

def MatmulOp : Demo_Op<"matmul"> {
  let summary = "matrix multiplication";
  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>:$lhs, AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>:$rhs);
  let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>);
  let builders = [
      OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs), [{
          $_state.addTypes(mlir::RankedTensorType::get({3, 4}, $_builder.getF32Type()));
          $_state.addOperands({lhs, rhs});
      }]>
  ];
}

def PrintOp : Demo_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins F64:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PrintVecOp : Demo_Op<"print_vec"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef, AnyVector]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}