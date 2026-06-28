#' @title Map embedding quality using a trained normalizing-flow density model
#' @name sits_flow_density_map
#' @author Alexandre Assuncao, \email{alexcarssuncao@@gmail.com}
#'
#' @description
#' Applies a trained \code{\link[sits]{sits_flow_density}} model spatially
#' across an embeddings cube, producing a \code{probs_cube} whose bands hold
#' per-class embedding-quality scores. The result is automatically plotted and
#' returned invisibly so that it can be passed to downstream \pkg{sits}
#' functions (e.g., \code{\link[sits]{sits_label_classification}},
#' \code{\link[sits]{sits_smooth}}).
#'
#' @section Motivation:
#' After encoding a raster data cube with \code{\link[sits]{sits_encode}} and
#' classifying the embedding cube with any \pkg{sits} model, it is useful to
#' ask: \emph{"How well do the embeddings of each pixel match the class they
#' were assigned to?"} A normalizing-flow model trained on labelled embeddings
#' answers this question by assigning a log-likelihood score to every embedding
#' vector under each class-conditional density.
#'
#' @section Scoring:
#' For a \strong{conditional} flow (default), the function evaluates
#'
#' \deqn{
#'   q_{p,c} = \log p_\theta(h_p \mid c)
#' }
#'
#' for every pixel \eqn{p} and every requested class \eqn{c}. Here \eqn{h_p}
#' is the embedding vector of pixel \eqn{p} extracted from \code{emb_cube}.
#' The result is a multi-band output with one band per class.
#'
#' For an \strong{unconditional} flow (\code{conditional = FALSE} at training
#' time), the function evaluates \eqn{\log p_\theta(h_p)} once per pixel and
#' replicates the result across all requested class bands.
#'
#' @section Normalization:
#' Because raw log-density values span a large negative range, each class band
#' is stored using a \strong{per-block percentile stretch}:
#'
#' \deqn{
#'   \tilde{q}_{p,c}
#'   = \frac{q_{p,c} - Q_{0.02}(q_{\cdot,c})}{Q_{0.98}(q_{\cdot,c}) -
#'     Q_{0.02}(q_{\cdot,c})},
#' }
#'
#' clipped to \eqn{[0, 1]} and then scaled to \code{[1, 10 000]} for
#' \code{INT2U} storage (following the \code{probs_cube} convention with scale
#' factor 0.0001). Accordingly:
#'
#' \itemize{
#'   \item Values near \strong{10 000} indicate that the pixel's embedding is
#'     \emph{highly typical} for that class (dense region of the class
#'     distribution).
#'   \item Values near \strong{1} indicate that the pixel's embedding is
#'     \emph{atypical} for that class (outlier or out-of-distribution region).
#' }
#'
#' @param flow_model Trained density model returned by
#'   \code{\link[sits]{sits_flow_density}} (an object of class
#'   \code{"sits_flow_density"}).
#' @param emb_cube Embeddings data cube (class \code{"raster_cube"}) produced
#'   by \code{\link[sits]{sits_encode}}. Each band corresponds to one dimension
#'   of the embedding space.
#' @param label_map Classified map (class \code{"class_cube"}) produced by
#'   applying \code{\link[sits]{sits_label_classification}} to the probability
#'   cube obtained from classifying \code{emb_cube}. Used to validate that the
#'   cube tiles match spatially. Must cover the same tiles as \code{emb_cube}.
#' @param classes Optional character vector of class names to visualize. Each
#'   name must appear in the set of classes used to train \code{flow_model}.
#'   When \code{NULL} (the default) all training classes are included.
#' @param roi Optional region of interest used to restrict processing and
#'   plotting. May be provided as a path to a polygon shapefile, an \code{sf}
#'   object, or a named bounding-box vector with elements \code{xmin},
#'   \code{xmax}, \code{ymin}, \code{ymax} (projected) or \code{lon_min},
#'   \code{lon_max}, \code{lat_min}, \code{lat_max} (WGS84).
#' @param memsize Integer. Memory available for processing in GB
#'   (minimum 1). Controls the block size used for chunk-parallel I/O.
#' @param multicores Integer. Number of CPU cores used for parallel block
#'   processing (minimum 1).
#' @param output_dir Character. Directory where output GeoTIFF files will be
#'   written. The directory must already exist.
#' @param version Character. Version tag appended to output file names (e.g.
#'   \code{"v1"}). Version strings are case-insensitive in \pkg{sits}.
#' @param verbose Logical. If \code{TRUE}, print per-tile timing information.
#' @param progress Logical. If \code{TRUE}, show a progress bar during block
#'   processing.
#' @param ... Additional arguments forwarded to \code{\link{plot.probs_cube}}
#'   (e.g., \code{palette}, \code{scale}, \code{legend_position}).
#'
#' @return A \code{probs_cube} (returned \strong{invisibly}) with one band per
#'   element of \code{classes}. Band values are normalized log-density scores in
#'   \code{[1, 10 000]}: higher values denote more typical embeddings for that
#'   class. The cube can be passed to \code{\link[sits]{sits_smooth}},
#'   \code{\link[sits]{sits_label_classification}}, or
#'   \code{\link[sits]{plot.probs_cube}} for further analysis.
#'
#' @note
#' The percentile stretch is computed \strong{within each processing block},
#' not globally across the entire tile. This means that the absolute numeric
#' values are not comparable across tiles or blocks, but the relative ordering
#' of pixels within a block is preserved. For applications that require
#' cross-tile comparability, consider applying
#' \code{\link[sits]{sits_smooth}} after calling this function.
#'
#' @seealso
#' \itemize{
#'   \item \code{\link[sits]{sits_flow_density}} — trains the normalizing-flow
#'     model used here.
#'   \item \code{\link[sits]{sits_encode}} — creates the \code{emb_cube} input.
#'   \item \code{\link[sits]{sits_label_classification}} — creates the
#'     \code{label_map} input.
#'   \item \code{\link[sits]{plot.probs_cube}} — underlying plot method; extra
#'     arguments in \code{...} are forwarded here.
#' }
#'
#' @examples
#' if (sits_run_examples()) {
#'     # --- 1. Create a regular cube and encode it ---------------------------
#'     data_dir <- system.file("extdata/raster/mod13q1", package = "sits")
#'     cube <- sits_cube(
#'         source     = "BDC",
#'         collection = "MOD13Q1-6.1",
#'         data_dir   = data_dir
#'     )
#'     enc <- sits_pre_train(
#'         samples        = samples_modis_ndvi,
#'         encoder_method = sits_mae(mask_ratio = 0.5)
#'     )
#'     emb_cube <- sits_encode(
#'         data       = cube,
#'         encoder    = enc,
#'         output_dir = tempdir()
#'     )
#'
#'     # --- 2. Classify the embedding cube to get a label map ---------------
#'     rfor_model <- sits_train(
#'         samples_modis_ndvi,
#'         sits_rfor()
#'     )
#'     # Train an embedding-space classifier on encoded training samples
#'     emb_samples <- sits_encode(data = samples_modis_ndvi, encoder = enc)
#'     emb_model   <- sits_train(emb_samples, sits_rfor())
#'     probs_cube  <- sits_classify(
#'         data       = emb_cube,
#'         ml_model   = emb_model,
#'         output_dir = tempdir()
#'     )
#'     label_map <- sits_label_classification(
#'         cube       = probs_cube,
#'         output_dir = tempdir()
#'     )
#'
#'     # --- 3. Train a flow density model on the encoded samples ------------
#'     flow_model <- sits_flow_density(emb_samples)
#'
#'     # --- 4. Map embedding quality ----------------------------------------
#'     density_cube <- sits_flow_density_map(
#'         flow_model = flow_model,
#'         emb_cube   = emb_cube,
#'         label_map  = label_map,
#'         classes    = c("Forest", "Pasture"),
#'         output_dir = tempdir()
#'     )
#' }
#'
#' @export
sits_flow_density_map <- function(flow_model,
                                   emb_cube,
                                   label_map,
                                   classes    = NULL,
                                   roi        = NULL,
                                   memsize    = 8L,
                                   multicores = 2L,
                                   output_dir,
                                   version    = "v1",
                                   verbose    = FALSE,
                                   progress   = TRUE,
                                   ...) {
    # ---- Input validation ---------------------------------------------------
    .check_set_caller("sits_flow_density_map")
    # Require torch (the flow model uses it internally)
    .check_require_packages("torch")
    # flow_model must be a trained sits_flow_density closure
    .check_that(
        inherits(flow_model, "sits_flow_density"),
        msg = paste0(
            "flow_model must be a trained sits_flow_density model ",
            "(returned by sits_flow_density())."
        )
    )
    # emb_cube must be a raster cube (embeddings_cube is a subclass)
    .check_is_raster_cube(emb_cube)
    # label_map must be a class_cube
    .check_is_class_cube(label_map)
    # Standard parameter checks
    .check_int_parameter(memsize,    min = 1L)
    .check_int_parameter(multicores, min = 1L)
    .check_output_dir(output_dir)
    # Version normalisation (sits uses lowercase version tags)
    version  <- .message_version(version)
    progress <- .message_progress(progress)
    verbose  <- .message_verbose(verbose)

    # ---- Extract metadata from the flow_model closure environment ----------
    flow_env    <- environment(flow_model)
    # Class labels used during training (character vector)
    labels      <- flow_env[["labels"]]
    # Boolean: was the model trained with class conditioning?
    conditional <- flow_env[["conditional"]]
    # Feature column names as they appear in the training predictor data frame.
    # For a single-timestep embedding tibble, .predictors() pivot_wider
    # appends the timestep index "1" to each band name (e.g. "E1" → "E11").
    train_set   <- flow_env[["train_set"]]
    feat_names  <- names(train_set)[-c(1L, 2L)]   # strip sample_id and label

    # ---- Resolve the set of classes to score --------------------------------
    if (.has(classes)) {
        # Validate that every requested class exists in the training labels
        unknown <- setdiff(classes, labels)
        .check_that(
            length(unknown) == 0L,
            msg = paste0(
                "The following classes were not seen during flow training: ",
                paste(unknown, collapse = ", "), "."
            )
        )
    } else {
        # Default: score all training classes
        classes <- labels
    }

    # ---- Validate tile-level spatial alignment -----------------------------
    # The label_map must contain at least all tiles present in emb_cube.
    emb_tiles   <- .cube_tiles(emb_cube)
    label_tiles <- .cube_tiles(label_map)
    missing_tiles <- setdiff(emb_tiles, label_tiles)
    .check_that(
        length(missing_tiles) == 0L,
        msg = paste0(
            "label_map is missing tiles that exist in emb_cube: ",
            paste(missing_tiles, collapse = ", "), "."
        )
    )

    # ---- Spatial filter (ROI) ----------------------------------------------
    if (.has(roi)) {
        roi      <- .roi_as_sf(roi)
        emb_cube <- .cube_filter_spatial(cube = emb_cube, roi = roi)
    }

    # ---- Extract embedding band names from the cube ------------------------
    emb_bands <- .cube_bands(emb_cube)

    # ---- Compute optimal block size ----------------------------------------
    # Mirror the block-sizing logic used in sits_classify.raster_cube.
    # npaths = number of embedding bands + number of output class bands.
    block <- .raster_file_blocksize(
        .raster_open_rast(.tile_path(emb_cube))
    )
    job_block_memsize <- .jobs_block_memsize(
        block_size  = .block_size(block = block, overlap = 0),
        npaths      = length(emb_bands) + length(classes),
        nbytes      = 8,
        proc_bloat  = .conf("processing_bloat")
    )
    multicores <- .jobs_max_multicores(
        job_block_memsize = job_block_memsize,
        memsize    = memsize,
        multicores = multicores
    )
    block <- .jobs_optimal_block(
        job_block_memsize = job_block_memsize,
        block      = block,
        image_size = .tile_size(.tile(emb_cube)),
        memsize    = memsize,
        multicores = multicores
    )

    # ---- Prepare parallel processing ----------------------------------------
    started <- .parallel_start(
        workers    = multicores,
        log        = verbose,
        output_dir = output_dir
    )
    if (started) {
        on.exit(.parallel_stop(), add = TRUE)
    }
    # Print overall start information (block dimensions, cores)
    start_time <- .classify_verbose_start(verbose, block)
    on.exit(.classify_verbose_end(verbose, start_time), add = TRUE)

    # ---- Process each tile -------------------------------------------------
    density_cube <- .cube_foreach_tile(emb_cube, function(emb_tile) {
        # Find the label_map tile that spatially matches this emb_tile
        tile_name  <- .tile_name(emb_tile)
        label_tile <- .cube_filter_tiles(label_map, tile_name)
        # Score the tile and return a probs_cube tile
        .flow_score_tile(
            emb_tile   = emb_tile,
            label_tile = label_tile,
            flow_model = flow_model,
            classes    = classes,
            emb_bands  = emb_bands,
            feat_names = feat_names,
            block      = block,
            roi        = roi,
            multicores = multicores,
            output_dir = output_dir,
            version    = version,
            verbose    = verbose,
            progress   = progress
        )
    })

    # ---- Plot the density cube with the requested classes ------------------
    p <- plot(density_cube, labels = classes, ...)
    print(p)

    # Return the probs_cube invisibly so the caller can chain operations
    invisible(density_cube)
}
