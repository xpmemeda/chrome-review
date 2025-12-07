func.func @main(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %C = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%A : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %C : tensor<4x4xf32>
}
