module ITensorGPU
using Adapt: adapt
using CUDA: CUDA, cu
export cu
using ITensors: cpu
export cpu

using ITensors: ITensor, cpu, cu, randomITensor
function cuITensor(args...; kwargs...)
  return adapt(CuArray, ITensor(args...; kwargs...))
end
function randomCuITensor(args...; kwargs...)
  return adapt(CuArray, randomITensor(args...; kwargs...))
end
export cuITensor, randomCuITensor

# TODO: Change over to `using ITensorMPS`
# once it is registered.
using ITensors.ITensorMPS: MPO, MPS, randomMPS
function cuMPS(args...; kwargs...)
  return adapt(CuArray, MPS(args...; kwargs...))
end
function productCuMPS(args...; kwargs...)
  return adapt(CuArray, MPS(args...; kwargs...))
end
function randomCuMPS(args...; kwargs...)
  return adapt(CuArray, randomMPS(args...; kwargs...))
end
function cuMPO(args...; kwargs...)
  return adapt(CuArray, MPO(args...; kwargs...))
end
cuMPO(tn::MPO) = cu(tn)
export cuMPO, cuMPS, productCuMPS, randomCuMPO, randomCuMPS
end
