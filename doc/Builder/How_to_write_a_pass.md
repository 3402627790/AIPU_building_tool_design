# How to write a pass for lowering

* IR to Target code: lowering
* How to write a pass by example

## IR to target code: lowering

The IR(Intermediate Representation) is a intermediate status from a program language to a machine code.  In our case, our IR is for describing the neural network computational graph. Our target code is the instructions of ZY.

It is not always that all IR operations have a corresponding instruction, generally an single IR operation need several machine instruction to finish. The lowering of builder is the process that transforms from a high level representation (IR operators) to a lower representation (ZY ISAs). The lowering is done by several transformation passes.


## How to write a pass by example

## Notes

* if your pass add/remove node, please confirm that the `origin_node` attribute is set.
* if your pass create a new TensorShape, please make sure the layout is well set.
* if you create a new shared memory Tensor, please put it in your node's `placehodler`
* if your pass add/remove node, please make sure the input/output tensor is same as your pass, i.e. you can NOT delete any created tensor, and make sure your pass final output is the original output.
