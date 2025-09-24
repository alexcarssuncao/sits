test_that("sits_tsne", {

    set.seed(123)

    models  <- c(sits_mlp, sits_tempcnn)#, sits_tae, sits_lighttae, sits_lstm_fcn)
    samples <- samples_modis_ndvi

    trained  <- lapply(models, function(m) sits_train(ml_method = m(epochs = 1), samples = samples_modis_ndvi))

    # baseline run: ignore duplicate embeddings
    tSNE_str <- lapply(trained, function(m)
        suppressWarnings(sits_tsne(model = m, samples = samples, perplexity = 30, rounds = 100))
    )

    # Structure checks
    check_tsne <- function(x) {
        expect_s3_class(x, "sits_tsne")
        expect_true(is.list(x$tsne))
        expect_true(is.vector(x$labels))
        expect_true(length(x$labels) <= nrow(samples))
    }

    lapply(tSNE_str, function(x) check_tsne(x))

    # clamping path: also allow test to pass whether it warns or not
    tSNE_clamp <- lapply(trained, function(m) {
        out <- NULL
        suppressWarnings(
            out <- sits::sits_tsne(model = m, samples = samples, perplexity = 999, rounds = 100)
        )
        check_tsne(out)

        # If Rtsne returns perplexity, assert it was clamped
        if (!is.null(out$tsne$perplexity)) {
            N_eff <- length(out$labels)
            max_perp <- floor((N_eff - 1L) / 3L)
            expect_true(out$tsne$perplexity <= max(5L, max_perp))
        }
        out
    })

})
