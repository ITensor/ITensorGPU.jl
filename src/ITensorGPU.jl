module ITensorGPU
using Adapt: adapt
using CUDA: CUDA, CuArray, cu
export cu
using ITensors: cpu
export cpu

using ITensors: ITensor, cpu, cu, random_itensor
function cuITensor(args...; kwargs...)
  return adapt(CuArray, ITensor(args...; kwargs...))
end
function randomCuITensor(args...; kwargs...)
  return adapt(CuArray, random_itensor(args...; kwargs...))
end
export cuITensor, randomCuITensor

using ITensorMPS: MPO, MPS, randomMPO, random_mps
function cuMPS(args...; kwargs...)
  return adapt(CuArray, MPS(args...; kwargs...))
end
cuMPS(tn::MPS) = adapt(CuArray, tn)
function productCuMPS(args...; kwargs...)
  return adapt(CuArray, MPS(args...; kwargs...))
end
function randomCuMPS(args...; kwargs...)
  return adapt(CuArray, random_mps(args...; kwargs...))
end
function cuMPO(args...; kwargs...)
  return adapt(CuArray, MPO(args...; kwargs...))
end
cuMPO(tn::MPO) = adapt(CuArray, tn)
function randomCuMPO(args...; kwargs...)
  return adapt(CuArray, randomMPO(args...; kwargs...))
end
export cuMPO, cuMPS, productCuMPS, randomCuMPO, randomCuMPS
end
