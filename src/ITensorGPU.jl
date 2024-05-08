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

# TODO: Change over to `using ITensorMPS`
# once it is registered.
using ITensors.ITensorMPS: MPO, MPS, randomMPS
function cuMPS(args...; kwargs...)
  return cu(MPS(args...; kwargs...))
end
function productCuMPS(args...; kwargs...)
  return cu(MPS(args...; kwargs...))
end
function randomCuMPS(args...; kwargs...)
  return cu(randomMPS(args...; kwargs...))
end
function cuMPO(args...; kwargs...)
  return cu(MPO(args...; kwargs...))
end
export cuMPO, cuMPS, productCuMPS, randomCuMPO, randomCuMPS
end
