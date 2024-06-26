if VERSION < v"1.8" && "@v#.#" ∉ LOAD_PATH
  push!(LOAD_PATH, "@v#.#")
end

using ITensorGPU, Test, CUDA

@testset "ITensorGPU.jl" begin
  @testset "Test exports" begin
    @test issetequal(
      names(ITensorGPU),
      [
        :ITensorGPU,
        :cpu,
        :cu,
        :cuITensor,
        :cuMPO,
        :cuMPS,
        :productCuMPS,
        :randomCuITensor,
        :randomCuMPO,
        :randomCuMPS,
      ],
    )
  end
  if !CUDA.has_cuda()
    println("System does not have CUDA, skipping test.")
  else
    println(
      "Running ITensorGPU tests with a runtime CUDA version: $(CUDA.runtime_version())"
    )

    CUDA.allowscalar(false)
    @testset "ITensorGPU.jl" begin
      @testset "$filename" for filename in (
        "test_dmrg.jl",
        "test_cuitensor.jl",
        "test_cudiag.jl",
        "test_cudense.jl",
        "test_cucontract.jl",
        "test_cumpo.jl",
        "test_cumps.jl",
      )
        println("Running $filename")
        include(filename)
      end
    end
  end
end
