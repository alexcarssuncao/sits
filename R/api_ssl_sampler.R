#' @title SSL sampling generic function
#'
#' @author Alexandre Assuncao \email{alexcarssuncao@@gmail.com}
#'
#' @description
#' Internal generic for creating self-supervised learning (SSL) sampling datasets.
#' Supports different SSL methods (e.g., triplet sampling for contrastive learning).
#'
#' @param samples A sits tibble with time series samples.
#' @param method SSL sampling method (e.g., "triplet", "simclr", "bert").
#' @param ... Additional parameters (depends on method).
#'
#' @return An SSL training dataset object (list or custom class).
#' @keywords internal
#' @noRd
.ssl_sampler <- function(samples, method = "semi_hard_triplet", ...) {
    UseMethod(".ssl_sampler", samples)
}


#' @keywords internal
#' @noRd
.ssl_sampler.sits <- function(samples, method = "semi_hard_triplet", ...) {
    if (method == "semi_hard_triplet") {
        .sits_semi_hard_triplet_sampler(samples, ...)
    } else {
        stop("Unsupported SSL sampling method: ", method)
    }
}


#' Batch‐Hard Triplet Sampler for SSL in SITS
#'
#' Implements batch‐hard triplet mining as described by Schroff et al. (2015).
#' For each mini‐batch, it:
#' 1. Samples \code{classes_per_batch} distinct classes
#' 2. Samples \code{samples_per_class} examples per class
#' 3. Computes all pairwise distances via \code{embed_fn} and \code{dist_fn}
#' 4. For each anchor, selects the hardest positive (furthest same‐class) and hardest negative (closest different‐class)
#'
#' @param samples A \code{tibble} or \code{data.frame} containing at least:
#'   \itemize{
#'     \item \code{time_series} — a list‐column of per‐sample time‐series (tibbles/data.frames)
#'     \item \code{label}       — a factor or character vector of class labels
#'   }
#' @param classes_per_batch Integer. Number of distinct classes to include in each mini‐batch.
#'   If \code{NULL}, defaults to \code{min(n_classes, 8)}.
#' @param samples_per_class Integer. Number of samples per class in each mini‐batch.
#'   If \code{NULL}, defaults to \code{max(2, floor(target_batch_size / classes_per_batch))}.
#' @param target_batch_size Integer. Approximate total batch size used to compute \code{samples_per_class}.
#'   Default is \code{64}.
#' @param num_triplets Integer. Total number of triplets to generate. Defaults to \code{nrow(samples)}.
#' @param embed_fn Function. Maps a single \code{time_series} tibble to a numeric embedding vector.
#' @param dist_fn Function. Computes a distance between two numeric vectors. Defaults to Euclidean distance.
#'
#' @return A \code{tibble} with columns:
#'   \describe{
#'     \item{\code{anchor}}{list of time‐series tibbles for the anchor}
#'     \item{\code{positive}}{list of time‐series tibbles for the positive}
#'     \item{\code{negative}}{list of time‐series tibbles for the negative}
#'     \item{\code{anchor_idx}}{integer index in \code{samples} of the anchor}
#'     \item{\code{positive_idx}}{integer index in \code{samples} of the positive}
#'     \item{\code{negative_idx}}{integer index in \code{samples} of the negative}
#'   }
#'
#' @details
#' This function implements the “batch‐hard” mining strategy of FaceNet:
#' within each mini‐batch it identifies the hardest positives (furthest within‐class)
#' and hardest negatives (closest out‐of‐class) for each anchor. This often yields
#' faster convergence and more discriminative embeddings than purely random triplets.
#'
#' @references
#' Schroff, F., Kalenichenko, D., & Philbin, J. (2015).
#' _FaceNet: A Unified Embedding for Face Recognition and Clustering_.
#' In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (CVPR), 815–823.
#' doi:10.1109/CVPR.2015.7298682
#'
#' @examples
#' \dontrun{
#' library(dplyr)
#'
#' samples <- tibble(
#'   label = c("A","A","B","B"),
#'   time_series = list(
#'     tibble(x=1:5,y=rnorm(5)),
#'     tibble(x=1:5,y=rnorm(5)),
#'     tibble(x=1:5,y=rnorm(5)),
#'     tibble(x=1:5,y=rnorm(5))
#'   )
#' )
#'
#' triplets <- .sits_ssl_triplet_sampler(
#'   samples,
#'   classes_per_batch  = NULL,
#'   samples_per_class = NULL,
#'   target_batch_size = 32
#' )
#' }
#'
#' @importFrom tibble   tibble
#' @importFrom purrr    map
#' @importFrom dplyr    bind_rows
#' @keywords internal
#' @noRd
.sits_semi_hard_triplet_sampler <- function(samples,                                      # sits samples
                                            classes_per_batch = NULL,                     # number of classes in each mini-batch
                                            samples_per_class = NULL,                     # number of sample instances in each mini-batch
                                            num_triplets = nrow(samples),                 # total triplets to make
                                            target_batch_size = 64,                       # default mini-batch size
                                            embed_fn = function(ts) {                     # a function to extract a fixed-length feature vector from each time_series
                                                # grab tibble's numeric columns
                                                mat <- as.matrix(ts[ , sapply(ts, is.numeric)])
                                                # flatten row‐wise
                                                as.vector(t(mat))
                                            },
                                            dist_fn = function(a, b) sqrt(sum((a - b)^2)) # distance function between two feature vectors
) {

    all_labels <- samples$label
    classes <- unique(all_labels)
    n_classes <- length(classes)
    triplets <- vector("list", length = 0)

    # 1) pick defaults if user didn't supply them
    if (is.null(classes_per_batch)) {
        classes_per_batch <- min(n_classes, 8)
    }
    if (is.null(samples_per_class)) {
        # scale target_batch_size using the number of classes
        # with at least two per class
        samples_per_class <- max(2, floor(target_batch_size / classes_per_batch))
    }

    while(length(triplets) < num_triplets) {
        # 1) Sample P classes and then K examples per class
        chosen_classes <- sample(classes, classes_per_batch)
        batch_idx <- unlist(
            purrr::map(chosen_classes, function(cls) {
                # find all dataset indices having label == cls
                idxs <- which(all_labels == cls)
                if (length(idxs) >= samples_per_class) {
                    # if there are at least K examples, pick K *distinct* ones
                    sample(idxs, samples_per_class)
                } else {
                    # if not enough, sample with replacement
                    sample(idxs, samples_per_class, replace = TRUE)
                }
            }))

        # 2) Build feature matrix and labels for batch
        feats <- purrr::map(batch_idx, ~ embed_fn(samples$time_series[[.x]]))
        feats <- do.call(rbind, feats)
        labs  <- all_labels[batch_idx]

        # 3) Pairwise distance matrix
        n <- nrow(feats)
        D <- matrix(0, n, n)

        for (i in seq_len(n)) {
            # compute distances from feats[i, ] to every row in feats
            D[i, ] <- vapply(seq_len(n),
                             function(j) dist_fn(feats[i, ], feats[j, ]),
                             numeric(1))
        }

        order_j  <- sample(seq_along(batch_idx))  # <-- shuffle anchor order

        # 4) For *each* anchor in this batch, pick hardest pos/neg
        for (j in order_j) {
            same_cls   <- which(labs == labs[j]  & seq_along(labs) != j)
            other_cls  <- which(labs != labs[j])
            if (length(same_cls)==0 || length(other_cls)==0) next

            # Select hardest negative
            neg_j <- other_cls[ which.min(D[j, other_cls]) ]
            d_neg <- D[j, neg_j]

            # distances to all same‐class examples
            d_pos_all <- D[j, same_cls]
            # keep only those strictly closer than the negative
            semi_idx <- same_cls[ d_pos_all < d_neg ]
            if (length(semi_idx) > 0) {
                # among those, pick the *hardest* semi‐hard positive
                pos_j <- semi_idx[ which.max(D[j, semi_idx]) ]
            } else {
                # fallback: no semi‐hards, pick the easiest positive
                pos_j <- same_cls[ which.min(d_pos_all) ]
            }

            triplets[[length(triplets) + 1]] <- tibble::tibble(
                anchor       = list(samples$time_series[[ batch_idx[j] ]]),
                positive     = list(samples$time_series[[ batch_idx[pos_j] ]]),
                negative     = list(samples$time_series[[ batch_idx[neg_j] ]]),
                anchor_idx   = batch_idx[j],
                positive_idx = batch_idx[pos_j],
                negative_idx = batch_idx[neg_j]
            )
            if (length(triplets) >= num_triplets) break
        }
    }
    dplyr::bind_rows(triplets[1:num_triplets])
}



.sits_random_triplet_sampler <- function(
        samples,                               # tibble: columns `time_series` (list), `label`
        num_triplets    = NULL,                # default: nrow(samples)
        skip_singletons = TRUE,                # avoid anchors from classes with only 1 sample
        seed            = NULL                 # optional reproducibility
) {
    if (!is.null(seed)) set.seed(seed)

    n <- nrow(samples)
    if (is.null(num_triplets)) num_triplets <- n

    # Prepare class→indices map
    labels <- samples$label
    if (is.factor(labels)) labels <- as.character(labels)
    idx_by_class <- split(seq_len(n), labels)
    classes <- names(idx_by_class)

    # Sanity checks
    if (length(classes) < 2L)
        stop("Need at least 2 classes to build (anchor, positive, negative) triplets.")

    # Anchor pool (optionally exclude singleton classes)
    class_sizes <- vapply(idx_by_class, length, integer(1))
    if (skip_singletons) {
        eligible_classes <- names(class_sizes[class_sizes >= 2L])
        anchor_pool <- unlist(idx_by_class[eligible_classes], use.names = FALSE)
        if (length(anchor_pool) == 0L)
            stop("All classes are singletons and skip_singletons=TRUE: no valid anchors.")
    } else {
        anchor_pool <- seq_len(n)
    }

    # Helpers
    sample_positive <- function(anchor_idx, pool) {
        if (length(pool) == 1L) {
            if (skip_singletons) return(NA_integer_)
            return(pool) # allow self-positive if desired
        }
        choices <- setdiff(pool, anchor_idx)
        if (length(choices) == 0L) {
            if (skip_singletons) return(NA_integer_)
            return(sample(pool, 1L, replace = TRUE))
        }
        sample(choices, 1L)
    }

    sample_negative <- function(anchor_label) {
        other <- setdiff(classes, anchor_label)
        if (length(other) == 0L) return(NA_integer_)
        neg_cls <- sample(other, 1L)
        sample(idx_by_class[[neg_cls]], 1L)
    }

    # Generate triplets
    triplets <- vector("list", num_triplets)
    wrote <- 0L
    max_tries <- max(10L * num_triplets, 10000L)  # guard against rare degenerate loops
    tries <- 0L

    while (wrote < num_triplets && tries < max_tries) {
        tries <- tries + 1L
        a <- sample(anchor_pool, 1L)
        a_lbl <- labels[a]

        pos <- sample_positive(a, idx_by_class[[a_lbl]])
        if (is.na(pos)) next

        neg <- sample_negative(a_lbl)
        if (is.na(neg)) next

        wrote <- wrote + 1L
        triplets[[wrote]] <- tibble::tibble(
            anchor       = list(samples$time_series[[a]]),
            positive     = list(samples$time_series[[pos]]),
            negative     = list(samples$time_series[[neg]]),
            anchor_idx   = a,
            positive_idx = pos,
            negative_idx = neg
        )
    }

    if (wrote == 0L)
        stop("No triplets could be formed (likely only singleton classes and skip_singletons=TRUE).")

    out <- dplyr::bind_rows(triplets[seq_len(wrote)])
    if (nrow(out) > num_triplets) out <- out[seq_len(num_triplets), ]
    out
}



