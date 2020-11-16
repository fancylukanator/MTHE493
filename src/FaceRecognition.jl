
module FaceRecognition

using Images
using FileIO
using StatsBase
using MultivariateStats

include("load_data.jl")
include("eigenfaces.jl")

export  load_images,
        Model,
        train_model,
        image_to_eigenfaces,
        eigenfaces_to_image,
        reconstruct_image,
        reconstruct_images,
        get_difference,
        get_eigenfaces,
        save,
        load

end
