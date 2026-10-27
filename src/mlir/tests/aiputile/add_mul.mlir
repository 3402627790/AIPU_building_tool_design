// Portable harness: source operations use their real public names in generic
// syntax. The full aipugc driver registers and verifies the source dialect.
module {
  func.func @kernel(%x: tensor<257xf32>, %b: tensor<257xf32>, %a: tensor<257xf32>) -> tensor<257xf32> {
    %t = "aipugc.add"(%x, %b) : (tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>
    %y = "aipugc.mul"(%t, %a) : (tensor<257xf32>, tensor<257xf32>) -> tensor<257xf32>
    return %y : tensor<257xf32>
  }
}
