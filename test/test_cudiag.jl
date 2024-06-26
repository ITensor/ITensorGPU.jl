using Combinatorics: permutations
using CUDA
using ITensorGPU
using ITensors
using ITensors.NDTensors: NDTensors
using LinearAlgebra: tr
using Test: @test, @test_broken, @testset

@testset "cuITensor $T Contractions" for T in (Float64, ComplexF64)
  mi, mj, mk, ml, ma = 2, 3, 4, 5, 6, 7
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  a = Index(ma, "a")
  @testset "Test contract cuITensors" begin
    Aij = cuITensor(random_itensor(T, i, j))
    Aji = cuITensor(random_itensor(T, j, i))
    Bij = cuITensor(random_itensor(T, i, j))
    Aik = cuITensor(random_itensor(T, i, k))
    Ajk = cuITensor(random_itensor(T, j, k))
    Ajl = cuITensor(random_itensor(T, j, l))
    Akl = cuITensor(random_itensor(T, k, l))
    Dv = rand(T, mi)
    D = itensor(ITensors.tensor(NDTensors.Diag(CuVector(Dv)), (i, i')))
    Ev = rand(T, mi)
    E = itensor(ITensors.tensor(NDTensors.Diag(CuVector(Ev)), (i, i'')))
    @testset "Test contract cuITensors (Matrix*Diag -> Matrix)" begin
      C = Aij * D
      @test_broken collect(CuArray(C)) ≈ collect(CuMatrix(Aij, j, i)) * diagm(0 => Dv)
    end
    @testset "Test contract cuDiagITensors (Diag*Diag -> Diag)" begin
      @test_broken E * D
      ## C = E * D
      ## cC = CuArray(C)
      ## @test collect(cC) ≈ diagm(0 => Ev) * diagm(0 => Dv)
    end
    @testset "Test contract cuDiagITensors (UniformDiag*Diag -> Diag)" begin
      scal = itensor(ITensors.tensor(NDTensors.Diag(2.0), (i, i'')))
      @test_broken scal * D
      ## C = scal * D
      ## @test collect(CuArray(C)) ≈ 2.0 .* diagm(0 => Dv)
      ## C = D * scal
      ## @test collect(CuArray(C)) ≈ 2.0 .* diagm(0 => Dv)
    end
    @testset "Test contract cuITensors (Matrix*UniformDiag -> Matrix)" begin
      scal = itensor(ITensors.tensor(NDTensors.Diag(T(2.0)), (i, i')))
      @test_broken scal * Aij
      ## C = scal * Aij
      ## @test cpu(C) ≈ 2.0 * cpu(replaceind(Aij, i, i')) atol = 1e-4
      ## C = Aij * scal
      ## @test_broken cpu(C) ≈ 2.0 * cpu(replaceind(permute(Aij, j, i), i, i')) atol = 1e-4
    end
  end # End contraction testset
end

@testset "cuITensor $T1, $T2 Contractions" for T1 in (Float64, ComplexF64),
  T2 in (Float64, ComplexF64)

  mi, mj, mk, ml, ma = 2, 3, 4, 5, 6, 7
  i = Index(mi, "i")
  j = Index(mj, "j")
  k = Index(mk, "k")
  l = Index(ml, "l")
  a = Index(ma, "a")
  @testset "Test contract cuITensors" begin
    Aij = cuITensor(random_itensor(T1, i, j))
    Aji = cuITensor(random_itensor(T1, j, i))
    Bij = cuITensor(random_itensor(T1, i, j))
    Dv = rand(T2, mi)
    D = itensor(ITensors.tensor(NDTensors.Diag(CuVector(Dv)), (i, i')))
    Ev = rand(T2, mi)
    E = itensor(ITensors.tensor(NDTensors.Diag(CuVector(Ev)), (i, i'')))
    @testset "Test contract cuITensors (Matrix*Diag -> Matrix)" begin
      C = Aij * D
      @test_broken CuArray(C)
      ## cC = CuArray(C)
      ## @test collect(cC) ≈ collect(CuMatrix(Aij, j, i)) * diagm(0 => Dv)
    end
    @testset "Test contract cuDiagITensors (Diag*Diag -> Diag)" begin
      @test_broken E * D
      ## C = E * D
      ## cC = CuArray(C)
      ## @test collect(cC) ≈ diagm(0 => Ev) * diagm(0 => Dv)
    end
    ## @testset "Test contract cuDiagITensors (UniformDiag*Diag -> Diag)" begin
    ##   scal = itensor(ITensors.tensor(NDTensors.Diag(T2(2.0)), (i, i'')))
    ##   C = scal * D
    ##   cC = CuArray(C)
    ##   @test collect(cC) ≈ 2.0 .* diagm(0 => Dv)
    ##   C = D * scal
    ##   cC = CuArray(C)
    ##   @test collect(cC) ≈ 2.0 .* diagm(0 => Dv)
    ## end
    ## @testset "Test contract cuITensors (Matrix*UniformDiag -> Matrix)" begin
    ##   scal = itensor(ITensors.tensor(NDTensors.Diag(T2(2.0)), (i, i')))
    ##   C = scal * Aij
    ##   cC = CuArray(C)
    ##   @test collect(cC) ≈ array(2.0 * cpu(replaceind(Aij, i, i'))) atol = 1e-4
    ##   C = Aij * scal
    ##   cC = CuArray(C)
    ##   @test_broken collect(cC) ≈ array(2.0 * cpu(replaceind(permute(Aij, j, i), i, i'))) atol =
    ##     1e-4
    ## end
  end # End contraction testset
end
