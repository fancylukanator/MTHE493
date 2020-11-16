
struct Model
    P::PCA{Float64}
    Z::ZScoreTransform
end

function train_model(images, d::Int; normalize::Bool)
    image_matrix = reduce(hcat, map(x->reshape(x, :, 1), convert.(Array{Float64}, images)))
    if normalize
        Z = fit(ZScoreTransform, image_matrix, dims=2)
        image_matrix = StatsBase.transform(Z, image_matrix)
        return Model(fit(PCA, image_matrix; maxoutdim=d), Z)
    end
    return fit(PCA, image_matrix; maxoutdim=d)
end

function image_to_eigenfaces(image, model::Model)
    return transform(model.P, StatsBase.transform(model.Z, reshape(convert(Array{Float64}, image), :, 1)))
end

function image_to_eigenfaces(image, P::PCA{Float64})
    return transform(P, reshape(convert(Array{Float64}, image), :, 1))
end

function eigenfaces_to_image(eigenfaces, M::Model)
    return Gray.(reshape(StatsBase.reconstruct(M.Z, reconstruct(M.P, eigenfaces)), 128, 128))
end

function eigenfaces_to_image(eigenfaces, P::PCA{Float64})
    return Gray.(reshape(reconstruct(P, eigenfaces), 128, 128))
end

function reconstruct_image(image::Array{Gray{N0f8},2}, model)
    return eigenfaces_to_image(image_to_eigenfaces(image, model), model)
end

function reconstruct_images(images, model)
    return [reconstruct_image(image, model) for image in images]
end

function get_difference(images, approximations)
    Gray.(abs.(reduce(hcat, images) .- reduce(hcat, approximations)))
end


function get_eigenfaces(model, d)
    eigenfaces = []
    for i = 1:d
        push!(eigenfaces, eigenfaces_to_image(reshape([Float64(i == j ? 100 : 0) for j in 1:d], d, 1), model))
    end
    return eigenfaces
end
