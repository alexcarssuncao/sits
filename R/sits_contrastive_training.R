#' @title Contrastive LTAE pre-training for sits
#' @name sits_contrastive_pretrain
#'
#' @description
#' Self-supervised pre-training using contrastive triplet loss and LTAE encoder.
#'
#' @param epochs Number of training epochs.
#' @param batch_size Batch size for training.
#' @param lr Learning rate.
#' @param margin Margin for triplet loss.
#' @param verbose Whether to print training progress.
#' @param log_every Period of epochs in which the model logs partial results
#'
#' @return A function for \code{\link[sits]{sits_pre_train}} (to be used as the \code{deep_method} parameter).
#'
#' @examples
#' if (sits_run_examples()) {
#'     model <- sits_pre_train(samples_modis_ndvi, sits_contrastive_ltae_pretrain())
#' }
#'
#' @export
sits_contrastive_ltae_pretrain <- function(epochs     = 100,
                                           batch_size = 32,
                                           lr         = 0.001,
                                           margin     = 1.0,
                                           verbose    = FALSE,
                                           log_every  = 5,
                                           seed       = NULL) {
    function(samples) {
        .sits_pre_train_contrastive_ltae(samples, epochs, batch_size, lr, margin, verbose, log_every, seed)
    }
}


.sits_pre_train_contrastive_ltae <- function(samples, epochs, batch_size, lr, margin, verbose, log_every, seed) {
    .check_samples(samples)
    # Get bands and timeline
    bands    <- .samples_bands(samples)
    timeline <- .samples_timeline(samples)
    n_bands  <- length(bands)
    n_times  <- .samples_ntimes(samples)
    # Data normalization
    ml_stats <- .samples_stats(samples)
    q02      <- as.numeric(ml_stats$q02)
    q98      <- as.numeric(ml_stats$q98)

    # ——————————————
    # 1) Triplet generation & DataLoader
    # ——————————————
    #triplets <- .sits_semi_hard_triplet_sampler(samples)
    triplets <- .sits_random_triplet_sampler(samples, num_triplets = 2 * nrow(samples), skip_singletons = FALSE)

    triplet_list <- purrr::map(seq_len(nrow(triplets)), function(i) {
        anchor_ts   <- triplets$anchor[[i]]
        positive_ts <- triplets$positive[[i]]
        negative_ts <- triplets$negative[[i]]
        list(
            anchor   = as.matrix(anchor_ts[ , -1]),
            positive = as.matrix(positive_ts[ , -1]),
            negative = as.matrix(negative_ts[ , -1])
        )
    })
    ds <- .TripletDataset(triplet_list)
    dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = TRUE)

    # to access this seed number from the model environment)
    torch_seed <- .torch_seed(seed)
    # Set torch seed
    torch::torch_manual_seed(torch_seed)
    # Define the TempCNN architecture
    tcnn_encoder <- torch::nn_module(
        classname = "tcnn_encoder",
        initialize = function(n_bands,
                              n_times,
                              kernel_sizes,
                              hidden_dims,
                              dropout_rates,
                              dense_layer_nodes,
                              dense_layer_dropout_rate) {

            self$hidden_dims <- hidden_dims
            self$embedding_dim <- dense_layer_nodes

            # first module - transform input to hidden dims
            self$conv_bn_relu1 <- .torch_conv1D_batch_norm_relu_dropout(
                input_dim    = n_bands,
                output_dim   = hidden_dims[[1L]],
                kernel_size  = kernel_sizes[[1L]],
                padding      = as.integer(kernel_sizes[[1L]] %/% 2L),
                dropout_rate = dropout_rates[[1L]]
            )
            # second module - 1D CNN
            self$conv_bn_relu2 <- .torch_conv1D_batch_norm_relu_dropout(
                input_dim    = hidden_dims[[1L]],
                output_dim   = hidden_dims[[2L]],
                kernel_size  = kernel_sizes[[2L]],
                padding      = as.integer(kernel_sizes[[2L]] %/% 2L),
                dropout_rate = dropout_rates[[2L]]
            )
            # third module - 1D CNN
            self$conv_bn_relu3 <- .torch_conv1D_batch_norm_relu_dropout(
                input_dim    = hidden_dims[[2L]],
                output_dim   = hidden_dims[[3L]],
                kernel_size  = kernel_sizes[[3L]],
                padding      = as.integer(kernel_sizes[[3L]] %/% 2L),
                dropout_rate = dropout_rates[[3L]]
            )
            # flatten 3D tensor to 2D tensor
            self$flatten <- torch::nn_flatten()
            # create a dense tensor
            self$dense <- .torch_linear_batch_norm_relu_dropout(
                input_dim    = hidden_dims[[3L]] * n_times,
                output_dim   = dense_layer_nodes,
                dropout_rate = dense_layer_dropout_rate
            )
        },
        forward = function(x) {
            # input is 3D n_samples x n_times x n_bands
            x <- x |>
                torch::torch_transpose(2L, 3L) |>
                self$conv_bn_relu1() |>
                self$conv_bn_relu2() |>
                self$conv_bn_relu3() |>
                self$flatten() |>
                self$dense()
        }
    )

    model <- tcnn_encoder(n_bands                  = n_bands,
                          n_times                  = n_times,
                          hidden_dims              = c(64L, 64L, 64L),
                          kernel_sizes             = c(3L, 3L, 3L),
                          dropout_rates            = c(0.20, 0.20, 0.20),
                          dense_layer_nodes        = 128L,
                          dense_layer_dropout_rate = 0.50
    )


    device    <- if (torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
    model     <- model$to(device = device)
    optimizer <- torch::optim_adam(model$parameters, lr = lr)

    # ——————————————
    # 3) Prepare logging containers
    # ——————————————
    model_log <- list(
        epoch         = integer(),
        mean_pos_dist = double(),
        mean_neg_dist = double(),
        embeddings    = list()
    )

    # ——————————————
    # 4) Training loop
    # ——————————————
    torch::torch_manual_seed(123)

    max_grad_norm <- 1.0

    for (epoch in seq_len(epochs)) {
        model$train()
        running_loss <- 0
        batch_count  <- 0

        coro::loop(for (batch in dl) {
            optimizer$zero_grad()

            anchor   <- batch$anchor$to(device = device)
            positive <- batch$positive$to(device = device)
            negative <- batch$negative$to(device = device)

            # tripwires on inputs
            if (as.logical(torch::torch_isnan(anchor)$any()$item()) ||
                as.logical(torch::torch_isinf(anchor)$any()$item()))
                stop("Non-finite values in anchor batch")

            z_anchor   <- model(anchor)
            z_positive <- model(positive)
            z_negative <- model(negative)

            # tripwires on embeddings
            if (as.logical(torch::torch_isnan(z_anchor)$any()$item()) ||
                as.logical(torch::torch_isinf(z_anchor)$any()$item()))
                stop("Non-finite values in z_anchor")

            loss <- .triplet_loss(z_anchor, z_positive, z_negative, margin)

            # tripwire on loss
            if (!is.finite(as.numeric(loss$item()))) stop("Loss became non-finite")

            loss$backward()

            # keep gradients in check
            torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = max_grad_norm)

            optimizer$step()

            running_loss <- running_loss + loss$item()
            batch_count  <- batch_count + 1
        })


        # Cosine LR (optionally keep a nonzero floor to avoid 0 exactly)
        min_lr <- lr * 1e-3
        if (epochs > 1) {
            cosw  <- 0.5 * (1 + cos(pi * (epoch - 1) / (epochs - 1)))
            new_lr <- min_lr + (lr - min_lr) * cosw
        } else {
            new_lr <- lr
        }
        optimizer$param_groups[[1]]$lr <- new_lr


        if (verbose) {
            cat(sprintf(
                "Epoch %2d | Avg Loss: %.4f | LR: %.2e\n",
                epoch,
                running_loss / batch_count,
                new_lr
            ))
        }
    }

    # ——————————————
    # 5) Wrap up & return
    # ——————————————

    # Serialize trained model
    serialized_model <- .torch_serialize_model(model)
    # Dummy predict function
    predict_fun <- function(x) {
        model <- .torch_unserialize_model(serialized_model)
        # Dummy: Return NA or embeddings if needed
        NA
    }

    # Helper function to extract torch model (for tsne)
    get_model <- function() {
        model <- .torch_unserialize_model(serialized_model)
        return(model)
    }

    # Attach the model as an attribute
    attr(predict_fun, "get_model") <- get_model

    # Set class for sits consistency
    predict_fun <- .set_class(predict_fun, "torch_model_contrastive_tcnn", "torch_model", "sits_model", class(predict_fun))

    return(predict_fun)
}


.TripletDataset <- torch::dataset(
    name = "TripletDataset",

    initialize = function(triplets, q02_band = NULL, q98_band = NULL,
                          clip = TRUE, eps = 1e-6) {
        self$triplets <- triplets
        self$q02_band <- q02_band
        self$q98_band <- q98_band
        self$clip <- clip
        self$eps  <- eps

        # If stats are provided, sanity-check lengths against band count
        if (!is.null(q02_band) || !is.null(q98_band)) {
            one <- triplets[[1]]$anchor
            one_mat <- as.matrix(one)
            n_bands <- ncol(one_mat)  # expects shape [n_times, n_bands] or [*, n_bands]
            stopifnot(length(q02_band) == n_bands, length(q98_band) == n_bands)
        }
    },

    .length = function() length(self$triplets),

    .getitem = function(index) {
        triplet <- self$triplets[[index]]

        normalize_mat <- function(mat) {
            # mat: [n_times, n_bands] (or any * x n_bands)
            if (is.null(self$q02_band)) return(mat)
            centered <- sweep(mat, 2, self$q02_band, `-`)
            denom    <- pmax(self$q98_band - self$q02_band, self$eps)
            scaled   <- sweep(centered, 2, denom, `/`)
            if (self$clip) scaled <- pmin(pmax(scaled, 0), 1)
            scaled
        }

        anchor_mat   <- normalize_mat(as.matrix(triplet$anchor))
        positive_mat <- normalize_mat(as.matrix(triplet$positive))
        negative_mat <- normalize_mat(as.matrix(triplet$negative))

        list(
            anchor   = torch::torch_tensor(anchor_mat),
            positive = torch::torch_tensor(positive_mat),
            negative = torch::torch_tensor(negative_mat)
        )
    }
)

#' @keywords internal
#' @noRd
.triplet_loss <- function(anchor, positive, negative, margin = 1.0) {
    pos_dist <- .pairwise_distance_squared(anchor, positive)
    neg_dist <- .pairwise_distance_squared(anchor, negative)
    loss <- torch::torch_clamp(pos_dist - neg_dist + margin, min = 0)
    loss$mean()
}

#' @keywords internal
#' @noRd
.pairwise_distance_squared <- function(x1, x2) {
    torch::torch_sum((x1 - x2) ^ 2, dim = 2)
}



