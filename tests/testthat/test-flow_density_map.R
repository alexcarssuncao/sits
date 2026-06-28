# ---- sits_flow_density_map() tests -----------------------------------------
#
# These tests exercise the full sits_flow_density pipeline:
#   sits_pre_train -> sits_encode (cube + samples) -> sits_flow_density
#   -> sits_classify -> sits_label_classification -> sits_flow_density_map
#
# The key regression being guarded is the *parallelism bug* that existed when
# sits_flow_density_map() called .parallel_start(), spawning idle PSOCK workers
# alongside the torch thread pool.  After the fix the function must:
#   * complete successfully with multicores = 1 (no workers ever started)
#   * complete successfully with multicores = 2 (no workers started either,
#     since block scoring is always sequential for torch models)
#   * leave no open sits parallel cluster on exit
#   * return a valid probs_cube with values in [1, 10 000]

# Helper: build the shared pipeline objects once per test file.
# We use a tiny MAE (5 epochs, 4-dim embedding, sits_tempcnn backbone) on the
# bundled MOD13Q1 sinop tile to keep wall-clock time manageable.
.flow_test_pipeline <- local({
    cache <- NULL
    function() {
        if (!is.null(cache)) {
            return(cache)
        }

        # ---- 1. Load small raster cube from package data --------------------
        data_dir <- system.file("extdata/raster/mod13q1", package = "sits")
        sinop <- sits_cube(
            source     = "BDC",
            collection = "MOD13Q1-6.1",
            data_dir   = data_dir,
            progress   = FALSE,
            verbose    = FALSE
        )

        # ---- 2. Train a tiny MAE encoder ------------------------------------
        mae <- sits_pre_train(
            samples        = samples_modis_ndvi,
            encoder_method = sits_mae(
                encoder_model = sits_tempcnn(),
                embedding_dim = 8L,
                mask_ratio    = 0.5,
                epochs        = 5L,
                verbose       = FALSE
            )
        )

        # ---- 3. Create output directory -------------------------------------
        out_dir <- file.path(tempdir(), "flow_density_map_test")
        dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

        # ---- 4. Encode cube and samples -------------------------------------
        emb_cube <- sits_encode(
            data       = sinop,
            encoder    = mae,
            memsize    = 4L,
            multicores = 1L,
            output_dir = file.path(out_dir, "emb_cube"),
            progress   = FALSE
        )

        emb_samples <- sits_encode(
            data    = samples_modis_ndvi,
            encoder = mae
        )

        # ---- 5. Classify embeddings -> probs -> labels ----------------------
        rfor <- sits_train(emb_samples, sits_rfor(num_trees = 30L))

        probs_cube <- sits_classify(
            data       = emb_cube,
            ml_model   = rfor,
            memsize    = 4L,
            multicores = 1L,
            output_dir = file.path(out_dir, "probs_cube"),
            progress   = FALSE
        )

        label_map <- sits_label_classification(
            cube       = probs_cube,
            output_dir = file.path(out_dir, "label_cube"),
            progress   = FALSE
        )

        # ---- 6. Train normalizing flow on encoded samples -------------------
        flow_model <- sits_flow_density(
            embeddings = emb_samples,
            epochs     = 5L,
            verbose    = FALSE
        )

        cache <<- list(
            emb_cube    = emb_cube,
            emb_samples = emb_samples,
            label_map   = label_map,
            flow_model  = flow_model,
            out_dir     = out_dir
        )
        cache
    }
})

# ---- parallelism regression: single core ------------------------------------
test_that("sits_flow_density_map: single core - no parallel cluster opened", {
    skip_if_not_installed("torch")
    skip_if_not_installed("luz")

    p <- .flow_test_pipeline()

    score_dir <- file.path(p$out_dir, "score_single")
    dir.create(score_dir, showWarnings = FALSE)

    # No PSOCK cluster should be running before the call
    expect_false(.parallel_is_open())

    density_cube <- sits_flow_density_map(
        flow_model = p$flow_model,
        emb_cube   = p$emb_cube,
        label_map  = p$label_map,
        classes    = sits_labels(p$emb_samples)[1L],
        multicores = 1L,
        memsize    = 4L,
        output_dir = score_dir,
        progress   = FALSE,
        verbose    = FALSE
    )

    # No cluster must have been left open after the call
    expect_false(.parallel_is_open())

    # Return value must be a probs_cube
    expect_s3_class(density_cube, "probs_cube")

    # Clean up
    unlink(score_dir, recursive = TRUE)
})

# ---- parallelism regression: multi-core -------------------------------------
test_that("sits_flow_density_map: multicores = 2 - no parallel cluster opened", {
    skip_if_not_installed("torch")
    skip_if_not_installed("luz")

    p <- .flow_test_pipeline()

    score_dir <- file.path(p$out_dir, "score_multi")
    dir.create(score_dir, showWarnings = FALSE)

    expect_false(.parallel_is_open())

    density_cube <- sits_flow_density_map(
        flow_model = p$flow_model,
        emb_cube   = p$emb_cube,
        label_map  = p$label_map,
        classes    = sits_labels(p$emb_samples)[1L],
        multicores = 2L,
        memsize    = 4L,
        output_dir = score_dir,
        progress   = FALSE,
        verbose    = FALSE
    )

    # The fix: no workers should have been created
    expect_false(.parallel_is_open())

    expect_s3_class(density_cube, "probs_cube")

    unlink(score_dir, recursive = TRUE)
})

# ---- output structure -------------------------------------------------------
test_that("sits_flow_density_map: output is a valid probs_cube", {
    skip_if_not_installed("torch")
    skip_if_not_installed("luz")

    p <- .flow_test_pipeline()

    classes_req <- sits_labels(p$emb_samples)[seq_len(2L)]
    score_dir   <- file.path(p$out_dir, "score_struct")
    dir.create(score_dir, showWarnings = FALSE)

    density_cube <- sits_flow_density_map(
        flow_model = p$flow_model,
        emb_cube   = p$emb_cube,
        label_map  = p$label_map,
        classes    = classes_req,
        multicores = 1L,
        memsize    = 4L,
        output_dir = score_dir,
        progress   = FALSE,
        verbose    = FALSE
    )

    # Correct class
    expect_s3_class(density_cube, "probs_cube")

    # One band per requested class
    expect_equal(length(sits_bands(density_cube)), length(classes_req))

    # Raster file exists and has the right number of layers
    rast_path <- density_cube$file_info[[1L]]$path[[1L]]
    expect_true(file.exists(rast_path))
    rast <- .raster_open_rast(rast_path)
    expect_equal(.raster_nlayers(rast), length(classes_req))

    # Values must be in [1, 10 000]  (INT2U probs_cube convention)
    vals <- .raster_get_values(rast)
    valid_vals <- vals[!is.na(vals)]
    expect_true(all(valid_vals >= 1L))
    expect_true(all(valid_vals <= 10000L))

    unlink(score_dir, recursive = TRUE)
})

# ---- class validation -------------------------------------------------------
test_that("sits_flow_density_map: unknown class raises an error", {
    skip_if_not_installed("torch")
    skip_if_not_installed("luz")

    p <- .flow_test_pipeline()

    expect_error(
        sits_flow_density_map(
            flow_model = p$flow_model,
            emb_cube   = p$emb_cube,
            label_map  = p$label_map,
            classes    = "DOES_NOT_EXIST",
            multicores = 1L,
            memsize    = 4L,
            output_dir = tempdir(),
            progress   = FALSE,
            verbose    = FALSE
        )
    )
})

# ---- all classes default (classes = NULL) -----------------------------------
test_that("sits_flow_density_map: classes = NULL scores all training labels", {
    skip_if_not_installed("torch")
    skip_if_not_installed("luz")

    p <- .flow_test_pipeline()

    score_dir <- file.path(p$out_dir, "score_all_classes")
    dir.create(score_dir, showWarnings = FALSE)

    density_cube <- sits_flow_density_map(
        flow_model = p$flow_model,
        emb_cube   = p$emb_cube,
        label_map  = p$label_map,
        classes    = NULL,
        multicores = 1L,
        memsize    = 4L,
        output_dir = score_dir,
        progress   = FALSE,
        verbose    = FALSE
    )

    n_train_labels <- length(sits_labels(p$emb_samples))
    expect_equal(length(sits_bands(density_cube)), n_train_labels)

    unlink(score_dir, recursive = TRUE)
})

# ---- cleanup shared temp files after all tests ------------------------------
withr::defer(
    unlink(file.path(tempdir(), "flow_density_map_test"), recursive = TRUE),
    envir = teardown_env()
)
