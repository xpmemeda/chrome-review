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
    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);
    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value), [{
            build($_builder, $_state, value.getType(), value);
        }]>,
        OpBuilder<(ins "double":$value)>
    ];
}

def AddOp : Tfcc_Op<"add"> {
    let summary = "element-wise addition operation";
    let description = [{+}];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def MulOp : Tfcc_Op<"mul"> {
    let summary = "element-wise multiplication operation";
    let description = [{*}];
    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}
