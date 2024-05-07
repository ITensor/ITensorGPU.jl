module ITensorGPU
using CUDA: CUDA
using ITensors: cpu, cu
export cpu, cu

using ITensors: ITensor, cpu, cu, randomITensor
function cuITensor(args...; kwargs...)
  return cu(ITensor(args...; kwargs...))
end
function randomCuITensor(args...; kwargs...)
  return cu(randomITensor(args...; kwargs...))
end
export cuITensor, randomCuITensor

using ITensors.ITensorMPS: MPO, MPS, randomMPS
function CuMPS(args...; kwargs...)
  return cu(MPS(args...; kwargs...))
end
function productCuMPS(args...; kwargs...)
  return cu(MPS(args...; kwargs...))
end
function randomCuMPS(args...; kwargs...)
  return cu(randomMPS(args...; kwargs...))
end
function CuMPO(args...; kwargs...)
  return cu(MPO(args...; kwargs...))
end
export cuMPO, cuMPS, productCuMPS, randomCuMPO, randomCuMPS
end
