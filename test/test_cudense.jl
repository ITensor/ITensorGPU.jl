using Combinatorics: permutations
using CUDA
using ITensorGPU
using ITensors
using LinearAlgebra: tr
using Test: @test, @testset

# gpu tests!
@testset "cuITensor, Dense{$SType} storage" for SType in (Float64, ComplexF64)
  mi, mj, mk, ml, ma = 2, 3, 4, 5, 6
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  a = Index(ma, "a")
  indices = [i, j, k, l, a]
  @testset "Test2 add CuDense" begin
    for i1 in indices, i2 in indices
      i1 == i2 && continue
      A = random_itensor(SType, i1, i2)
      B = random_itensor(SType, i1, i2)
      cuA = cu(A)
      cuB = cu(B)
      C = A + B
      cuC = cuA + cuB
      @test C ≈ cpu(cuC) #move to CPU to avoid scalar indexing error on GPU
      @test A ≈ cpu(cuA) #check does operation `+` modify cuA
      @test B ≈ cpu(cuB) #check does operation `+` modify cuB
      cuA += cuB
      @test cuC ≈ cuA
      @test B ≈ cpu(cuB) #check does operation `+=`` modify cuB
    end
  end
  @testset "Test2 subtract CuDense" begin
    for i1 in indices, i2 in indices
      i1 == i2 && continue
      A = random_itensor(SType, i1, i2)
      B = random_itensor(SType, i1, i2)
      cuA = cu(A)
      cuB = cu(B)
      C = A - B
      cuC = cuA - cuB
      @test C ≈ cpu(cuC) #move to CPU to avoid scalar indexing error on GPU
      @test A ≈ cpu(cuA) #check does operation `-` modify cuA
      @test B ≈ cpu(cuB) #check does operation `-` modify cuB
      cuA -= cuB
      @test cuC ≈ cuA
      @test B ≈ cpu(cuB) #check does operation `-=`` modify cuB
      #end
    end
  end
end # End Dense storage test
