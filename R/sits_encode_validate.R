

sits_validate_encode <- function(samples,
                                 encoder) {
    # set caller for error msg
    .check_set_caller("sits_validate_encode")
    # Verifies if 'torch' and 'luz' packages is installed
    .check_require_packages(c("torch", "luz"))
    # Check formals
    .check_sits_encode_validate(samples, encoder)

    # Gather samples' metadata
    n_samples <- nrow(samples)
    n_times <- .samples_ntimes(samples)
    bands <- .samples_bands(samples)
    n_bands <- length(bands)
    labels <- as.character(samples[["label"]])

    # Prepare and normalize input samples
    pred <- .predictors(samples)
    ml_stats <- .samples_stats(samples)
    pred <- .pred_normalize(pred, ml_stats)

    # Convert to tensor
    x <- array(
        as.matrix(.pred_features(pred)),
        dim = c(n_samples, n_times, n_bands)
    )
    x <- torch::torch_tensor(x, dtype = torch::torch_float())

    # Get reconstructed time series
    torch::with_no_grad(
        recon <- .ml_model(encoder)$model(x)
    )

    #--------- PLOTTING
    pred <- .predictors(samples)
    recon_arr <- as.array(recon)
    dim(recon_arr) <- c(n_samples, n_times * n_bands)
    .pred_features(pred) <- recon_arr
    recon_ts <- .pred_as_ts(pred, bands, .samples_timeline(samples))
    recon_samples <- samples
    .ts(recon_samples) <- recon_ts

    # Making sure the ts shapes match
    if (!all(recon$shape == x$shape)) {
        stop(.conf("messages", "sits_encode_validate_shape"))
    }

    diff_x <- x[, 2:n_times, ] - x[, 1:(n_times - 1), ]
    diff_recon <- recon[, 2:n_times, ] - recon[, 1:(n_times - 1), ]

    tensor_stats <- function(original, reconstructed) {

        stats <- list()

        # Squared error: [samples, time, bands]
        sq_error <- (reconstructed - original)^2
        # Global MSE over all samples, times and bands
        stats$mse_global <- sq_error$mean()$item()
        # MSE per band: average over samples and time
        mse_by_band <- as.numeric(sq_error$mean(dim = c(1, 2)))

        stats$band_stats <- tibble::tibble(
            band = bands,
            mse = mse_by_band
        )

        # Optional per-class MSE summary
        if (length(unique(samples$label)) > 1) {
            label <- NULL
            samples$mse <- as.numeric(sq_error$mean(dim = c(2, 3)))
            stats$mse_by_class <- samples |>
                dplyr::group_by(label) |>
                dplyr::summarise(
                    n = dplyr::n(),
                    mse_mean = mean(mse, na.rm = TRUE),
                    mse_sd = stats::sd(mse, na.rm = TRUE),
                    mse_min = min(mse, na.rm = TRUE),
                    mse_max = max(mse, na.rm = TRUE),
                    .groups = "drop"
                )
        }

       stats
    }

    result <- list(
        samples = samples,
        recon_samples = recon_samples,
        residuals = tensor_stats(x, recon),
        differences = tensor_stats(diff_x, diff_recon)
    )
    class(result) <- c("reconstruction_stats", class(result))

    result
}
