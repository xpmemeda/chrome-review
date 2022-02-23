include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tfcc_Dialect : Dialect {
    let name = "tfcc";
    let cppNamespace = "::mlir::tfcc";
}

class Tfcc_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Tfcc_Dialect, mnemonic, traits>;

def ConstantOp : Tfcc_Op<"constant", [NoSideEffect]> {
    let summary = "constant";
    let description = [{...}];
    let arguments = (ins F64Attr:$value);
    let results = (outs F64);
    let builders = [
        OpBuilder<(ins "FloatAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,
        OpBuilder<(ins "double":$value)>
    ];
}

def ReturnOp : Tfcc_Op<"return", [NoSideEffect, HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operation takes an no tensor operand and produces no results.
    }];
    let arguments = (ins);
}

def AddOp : Tfcc_Op<"add"> {
    let summary = "element-wise addition operation";
    let description = [{+}];
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def MulOp : Tfcc_Op<"mul"> {
    let summary = "element-wise multiplication operation";
    let description = [{*}];
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def PrintOp : Tfcc_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins F64:$input);

//   let assemblyFormat = "$input attr-dict `:` type($input)";
}