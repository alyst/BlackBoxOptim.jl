facts("Bimodal Cauchy Distributions") do

  context("sample bimodal cauchy with truncation above 1") do
    bc = BlackBoxOptim.BimodalCauchy(0.65, 0.1, 1.0, 0.1, clampBelow0=false)

    n_ok = 0
    for _ in 1:10000
      v = rand(bc)
      (0.0 < v <= 1.0) && (n_ok += 1)
    end
    @fact n_ok --> 10000
  end

  context("sample bimodal cauchy with truncation below 0 and above 1") do
    bc = BlackBoxOptim.BimodalCauchy(0.1, 0.1, 0.95, 0.1)

    n_ok = 0
    for _ in 1:10000
      v = rand(bc)
      (0.0 <= v <= 1.0) && (n_ok += 1)
    end
    @fact n_ok --> 10000
  end

end
