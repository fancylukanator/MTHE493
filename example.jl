
using Pkg
Pkg.activate("./")
Pkg.instantiate()
using FaceRecognition

#%%
#   generate_exampleS("./data/images/", 1000, 50, 1001:1010, "example1")

function generate_example(
      image_dir::AbstractString,
      n::Int,
      d::Int,
      test_range::UnitRange{Int};
      filename::AbstractString,
      save_eigenfaces::Bool=false
)
      training_images = load_images(image_dir, n)
      test_images = load_images(image_dir, test_range)

      model = train_model(training_images, d; normalize=true)

      reconstructed_images = reconstruct_images(test_images, model)
      example = vcat(
            reduce(hcat, test_images),
            reduce(hcat, reconstructed_images),
            get_difference(test_images, reconstructed_images).*3
      )
      save(string("./examples/", filename, ".png"),  example)
      if save_eigenfaces
            eigenfaces = reduce(vcat, (reduce(hcat, get_eigenfaces(model, d))[:, i*128*10+1:(i+1)*128*10] for i in 0:(d ÷ 10)-1))
            save(string("./examples/eigenfaces.png"), eigenfaces)
      end
      return example
end

#generate_example("./data/images/", 1000, 50, 1001:1010, "example2")

#%%
