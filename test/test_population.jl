facts("Population") do

  context("FloatVectorPopulation constructor") do
    p1 = FloatVectorPopulation(10, 2)
    @fact typeof(p1) => FloatVectorPopulation

    @fact size(p1.individuals, 1) => 10
    @fact size(p1.individuals, 2) => 2

    @fact size(p1.fitness, 1) => 10
    @fact size(p1.fitness, 2) => 2 # This is a bug since we cannot now the size of the fitness vectors before we have evaluated one...

    @fact size(p1.top, 1) => 10
    @fact size(p1.top, 2) => 2

    @fact size(p1.top_fitness, 1) => 10
    @fact size(p1.top_fitness, 2) => 2 # This is a bug since we cannot now the size of the fitness vectors before we have evaluated one...
  end

end