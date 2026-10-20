# Builds test/trim/app.jl with `juliac --trim=safe` and runs the resulting executable.
# Usage: julia --project=test/trim -e "using Pkg; Pkg.instantiate()" && julia --project=test/trim test/trim/build_and_run.jl

using Test, JuliaC

@testset "juliac --trim=safe" begin
    mktempdir() do dir
        image = ImageRecipe(output_type = "--output-exe", file = joinpath(@__DIR__, "app.jl"),
            trim_mode = "safe", project = @__DIR__)
        compile_products(image)
        exe = joinpath(dir, "app")
        link_products(LinkRecipe(image_recipe = image, outname = exe))
        @test readlines(`$exe`) == ["3.1", "3.1"]
    end
end
