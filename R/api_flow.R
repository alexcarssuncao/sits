#' Dataset for training normalizing flows on embedding vectors
#'
#' Internal torch dataset used by `sits_flow_density()` to train a
#' normalizing-flow density model over learned embeddings. The expected input
#' is a numeric matrix or array whose rows are samples and whose columns are
#' embedding dimensions.
#'
#' This dataset is designed for the embedding-quality workflow:
#'
#' \deqn{
#'   x \longmapsto h \in \mathbb{R}^d,
#' }
#'
#' where `h` is an embedding produced by a previous encoder, for example a
#' masked autoencoder, contrastive encoder, temporal CNN, or attention-based
#' encoder. The flow model then estimates either an unconditional density
#'
#' \deqn{
#'   p(h),
#' }
#'
#' or a class-conditional density
#'
#' \deqn{
#'   p(h \mid y).
#' }
#'
#' In the conditional case, the dataset returns both the embedding vector and
#' its class label to the model. The label is used as conditioning information
#' inside the affine coupling layers. In the unconditional case, the dataset
#' returns the embedding vector and a dummy target, because `luz` expects each
#' dataset item to have both an input and a target.
#'
#' This dataset does not create positive/negative pairs and does not split the
#' embedding vector. The split used by Real NVP-style affine coupling layers is
#' performed inside `.flow_coupling_layer()` using a binary mask. Thus each item
#' returned by this dataset corresponds to one embedding vector:
#'
#' \deqn{
#'   h_i = (h_{i1}, \ldots, h_{id}).
#' }
#'
#' @section Conditional output format:
#' When `conditional = TRUE`, `.getitem()` returns:
#'
#' \preformatted{
#' list(
#'   x = list(
#'     h = embedding_tensor,
#'     label = class_tensor
#'   ),
#'   y = class_tensor
#' )
#' }
#'
#' The model's `forward()` method receives `x`, extracts `h` and `label`, and
#' computes:
#'
#' \deqn{
#'   \log p_\theta(h \mid y).
#' }
#'
#' The target `y` is not used for classification. It is present for compatibility
#' with the `luz` training interface and may also be used by metrics or callbacks
#' if needed.
#'
#' @section Unconditional output format:
#' When `conditional = FALSE`, `.getitem()` returns:
#'
#' \preformatted{
#' list(
#'   x = embedding_tensor,
#'   y = dummy_integer_tensor
#' )
#' }
#'
#' The model computes:
#'
#' \deqn{
#'   \log p_\theta(h).
#' }
#'
#' @param x Numeric matrix or array with shape `[n_samples, embedding_dim]`.
#'   Each row is an embedding vector.
#' @param y Optional integer vector of class labels. Required when
#'   `conditional = TRUE`. Labels are expected to be encoded as integer class
#'   IDs compatible with `torch::nn_embedding()`.
#' @param conditional Logical. If `TRUE`, the dataset returns labels as part of
#'   the model input so that the flow estimates `p(h | y)`. If `FALSE`, labels
#'   are ignored and the flow estimates `p(h)`.
#'
#' @return A torch dataset object whose items are compatible with
#'   `.flow_density_net()`.
#'
#' @keywords internal
.flow_embedding_dataset <- torch::dataset(
    name = "flow_embedding_dataset",

    initialize = function(x, y = NULL, conditional = TRUE) {
        self$x <- torch::torch_tensor(x, dtype = torch::torch_float())
        self$conditional <- conditional

        if (!is.null(y)) {
            self$y <- torch::torch_tensor(y, dtype = torch::torch_long())
        } else {
            self$y <- NULL
        }
    },

    .getitem = function(i) {
        if (self$conditional) {
            list(
                x = list(
                    h = self$x[i, ],
                    label = self$y[i]
                ),
                y = self$y[i]
            )
        } else {
            list(
                x = self$x[i, ],
                y = torch::torch_tensor(0L, dtype = torch::torch_long())
            )
        }
    },

    .length = function() {
        self$x$size(1)
    }
)

#' Real NVP-style affine coupling layer for embedding density estimation
#'
#' Internal torch module implementing one affine coupling layer, following the
#' Real NVP construction of Dinh, Sohl-Dickstein, and Bengio. The layer defines
#' an invertible transformation between an embedding-space vector `x` and a
#' latent-space vector `z`.
#'
#' The layer is used in the density-evaluation direction
#'
#' \deqn{
#'   x \mapsto z,
#' }
#'
#' where `x` is a learned embedding and `z` is a latent vector under a simple
#' base distribution, typically a standard multivariate Gaussian:
#'
#' \deqn{
#'   z \sim \mathcal{N}(0, I).
#' }
#'
#' The layer also implements the inverse direction
#'
#' \deqn{
#'   z \mapsto x,
#' }
#'
#' which can be used for sampling or debugging the learned transformation.
#'
#' @section Affine coupling transform:
#' Let
#'
#' \deqn{
#'   x \in \mathbb{R}^d
#' }
#'
#' be an embedding vector, and let
#'
#' \deqn{
#'   m \in \{0,1\}^d
#' }
#'
#' be a fixed binary mask. The mask splits the embedding coordinates into two
#' groups. Coordinates where `m_j = 1` are kept fixed by this coupling layer.
#' Coordinates where `m_j = 0` are transformed.
#'
#' The masked vector is
#'
#' \deqn{
#'   x_m = x \odot m,
#' }
#'
#' where `\odot` denotes elementwise multiplication.
#'
#' A conditioner network computes scale and translation vectors:
#'
#' \deqn{
#'   (s, t) = g_\theta(x_m),
#' }
#'
#' or, in the conditional case,
#'
#' \deqn{
#'   (s, t) = g_\theta(x_m, e_y),
#' }
#'
#' where `e_y` is a learned class embedding for label `y`.
#'
#' The forward transformation, used for density evaluation, is:
#'
#' \deqn{
#'   z =
#'   m \odot x
#'   +
#'   (1 - m) \odot \left[(x - t) \odot \exp(-s)\right].
#' }
#'
#' Coordinate-wise, this means:
#'
#' \deqn{
#'   z_j = x_j
#'   \quad \text{if } m_j = 1,
#' }
#'
#' and
#'
#' \deqn{
#'   z_j = (x_j - t_j)\exp(-s_j)
#'   \quad \text{if } m_j = 0.
#' }
#'
#' The inverse transformation is:
#'
#' \deqn{
#'   x =
#'   m \odot z
#'   +
#'   (1 - m) \odot \left[z \odot \exp(s) + t\right].
#' }
#'
#' This is invertible because the conditioner depends only on the masked part,
#' which is left unchanged by the transformation. Therefore the same scale and
#' translation can be recomputed in both directions.
#'
#' @section Log-determinant:
#' The Jacobian of an affine coupling layer is triangular. Therefore its
#' determinant is cheap to compute. For the forward transformation
#' `x -> z`, the log absolute determinant is:
#'
#' \deqn{
#'   \log \left|
#'   \det \frac{\partial z}{\partial x}
#'   \right|
#'   =
#'   -\sum_j s_j.
#' }
#'
#' For the inverse transformation `z -> x`, the log absolute determinant is:
#'
#' \deqn{
#'   \log \left|
#'   \det \frac{\partial x}{\partial z}
#'   \right|
#'   =
#'   \sum_j s_j.
#' }
#'
#' This tractable Jacobian determinant is the central computational advantage of
#' Real NVP-style coupling layers.
#'
#' @section Conditional density:
#' When `conditional = TRUE`, the coupling layer learns class-conditioned scale
#' and translation functions:
#'
#' \deqn{
#'   s = s_\theta(x_m, y),
#'   \qquad
#'   t = t_\theta(x_m, y).
#' }
#'
#' The label `y` is first mapped to a trainable embedding vector:
#'
#' \deqn{
#'   e_y \in \mathbb{R}^{r},
#' }
#'
#' where `r = label_dim`. The conditioner input is the concatenation:
#'
#' \deqn{
#'   [x_m, e_y].
#' }
#'
#' This allows the full flow to estimate class-conditional densities:
#'
#' \deqn{
#'   p_\theta(x \mid y),
#' }
#'
#' which is useful for sample-quality assessment: a sample can be assigned a low
#' likelihood if its embedding is unusual for its declared class.
#'
#' @section Numerical stabilization:
#' The raw scale output is bounded using:
#'
#' \deqn{
#'   s \leftarrow c \tanh(s / c),
#' }
#'
#' where `c = clamp`. This prevents very large values of `exp(s)` and
#' `exp(-s)`, improving numerical stability during training.
#'
#' @param embedding_dim Integer. Dimension `d` of the embedding vectors.
#' @param hidden_dim Integer. Width of the hidden layers in the conditioner MLP.
#' @param mask Numeric vector of zeros and ones with length `embedding_dim`.
#'   Coordinates with mask value `1` are kept fixed; coordinates with mask value
#'   `0` are transformed.
#' @param n_labels Integer or `NULL`. Number of class labels. Required when
#'   `conditional = TRUE`.
#' @param conditional Logical. If `TRUE`, condition the coupling layer on class
#'   labels through a trainable label embedding.
#' @param label_dim Integer. Dimension of the trainable class embedding used when
#'   `conditional = TRUE`.
#' @param clamp Numeric. Bound applied to the scale vector through
#'   `clamp * tanh(s / clamp)`.
#'
#' @return A torch module with methods:
#' \describe{
#'   \item{`conditioner(x, y = NULL)`}{Computes masked scale and translation
#'   vectors.}
#'   \item{`forward(x, y = NULL)`}{Maps data-space embeddings to latent-space
#'   vectors and returns `list(z, log_det)`.}
#'   \item{`inverse(z, y = NULL)`}{Maps latent-space vectors back to
#'   embedding-space vectors and returns `list(x, log_det)`.}
#' }
#'
#' @references
#' Dinh, L., Sohl-Dickstein, J., & Bengio, S. Density estimation using Real NVP.
#' International Conference on Learning Representations, 2017.
#'
#' @keywords internal
.flow_coupling_layer <- torch::nn_module(
    "flow_coupling_layer",

    initialize = function(embedding_dim,
                          hidden_dim = 128L,
                          mask,
                          n_labels = NULL,
                          conditional = TRUE,
                          label_dim = 16L,
                          clamp = 2.0) {

        self$embedding_dim <- embedding_dim
        self$conditional <- conditional
        self$clamp <- clamp

        self$register_buffer(
            "mask",
            torch::torch_tensor(mask, dtype = torch::torch_float())
        )

        cond_dim <- embedding_dim

        if (conditional) {
            # Use n_labels + 1L so that sits' 1-indexed class codes
            # (seq_along(labels) = 1 … n_labels) are within the valid
            # range [0, num_embeddings - 1] of torch::nn_embedding.
            self$label_emb <- torch::nn_embedding(
                num_embeddings = n_labels + 1L,
                embedding_dim = label_dim
            )
            cond_dim <- cond_dim + label_dim
        }

        self$net <- torch::nn_sequential(
            torch::nn_linear(cond_dim, hidden_dim),
            torch::nn_relu(),
            torch::nn_batch_norm1d(hidden_dim),
            torch::nn_linear(hidden_dim, hidden_dim),
            torch::nn_relu(),
            torch::nn_batch_norm1d(hidden_dim),
            torch::nn_linear(hidden_dim, 2L * embedding_dim)
        )
    },

    conditioner = function(x, y = NULL) {
        x_masked <- x * self$mask

        if (self$conditional) {
            y_emb <- self$label_emb(y)
            input <- torch::torch_cat(list(x_masked, y_emb), dim = 2)
        } else {
            input <- x_masked
        }

        st <- self$net(input)
        chunks <- st$chunk(2L, dim = 2)

        s <- chunks[[1]]
        t <- chunks[[2]]

        # Keep scale stable.
        s <- self$clamp * torch::torch_tanh(s / self$clamp)

        # Only transform unmasked part.
        inv_mask <- 1 - self$mask
        s <- s * inv_mask
        t <- t * inv_mask

        list(s = s, t = t)
    },

    # data h -> base z
    forward = function(x, y = NULL) {
        pars <- self$conditioner(x, y)
        s <- pars$s
        t <- pars$t

        inv_mask <- 1 - self$mask

        z <- x * self$mask + inv_mask * ((x - t) * torch::torch_exp(-s))
        log_det <- -s$sum(dim = 2)

        list(z = z, log_det = log_det)
    },

    # base z -> data h
    inverse = function(z, y = NULL) {
        pars <- self$conditioner(z, y)
        s <- pars$s
        t <- pars$t

        inv_mask <- 1 - self$mask

        x <- z * self$mask + inv_mask * (z * torch::torch_exp(s) + t)
        log_det <- s$sum(dim = 2)

        list(x = x, log_det = log_det)
    }
)

#' Real NVP-style normalizing flow for embedding density estimation
#'
#' Internal torch module implementing a stack of affine coupling layers for
#' density estimation over learned embedding vectors. The model is intended for
#' use by `sits_flow_density()` to estimate either an unconditional embedding
#' density
#'
#' \deqn{
#'   p_\theta(h),
#' }
#'
#' or a class-conditional embedding density
#'
#' \deqn{
#'   p_\theta(h \mid y).
#' }
#'
#' Here `h` is an embedding vector produced by an encoder and `y` is an optional
#' class label. The model follows the Real NVP principle of composing invertible
#' affine coupling transformations with tractable Jacobian determinants.
#'
#' @section Flow construction:
#' Let
#'
#' \deqn{
#'   h \in \mathbb{R}^d
#' }
#'
#' be an embedding vector. The flow defines an invertible transformation
#'
#' \deqn{
#'   z = f_\theta(h; y),
#' }
#'
#' where `z` is a latent vector modeled by a standard Gaussian base density:
#'
#' \deqn{
#'   p_Z(z) = \mathcal{N}(z; 0, I).
#' }
#'
#' The transformation is a composition of `K = n_layers` affine coupling layers:
#'
#' \deqn{
#'   z =
#'   f_K \circ f_{K-1} \circ \cdots \circ f_1(h).
#' }
#'
#' Each coupling layer leaves a subset of coordinates unchanged and transforms
#' the complementary subset. Consecutive layers use alternating masks so that all
#' coordinates can be transformed across the full flow.
#'
#' @section Change of variables:
#' The model evaluates log-density using the change-of-variables formula:
#'
#' \deqn{
#'   \log p_\theta(h \mid y)
#'   =
#'   \log p_Z(z)
#'   +
#'   \log
#'   \left|
#'   \det
#'   \frac{\partial z}{\partial h}
#'   \right|.
#' }
#'
#' Since the transformation is a composition of coupling layers, the total
#' log-determinant is the sum of the log-determinants from each layer:
#'
#' \deqn{
#'   \log p_\theta(h \mid y)
#'   =
#'   \log p_Z(z)
#'   +
#'   \sum_{k=1}^{K}
#'   \log
#'   \left|
#'   \det
#'   \frac{\partial f_k}{\partial h_{k-1}}
#'   \right|.
#' }
#'
#' In the unconditional case, the same formula is used without `y`:
#'
#' \deqn{
#'   \log p_\theta(h)
#'   =
#'   \log p_Z(z)
#'   +
#'   \sum_{k=1}^{K}
#'   \log
#'   \left|
#'   \det
#'   J_k
#'   \right|.
#' }
#'
#' @section Base density:
#' The base density is a standard multivariate Gaussian. For each latent vector
#' `z`, the log-density is:
#'
#' \deqn{
#'   \log p_Z(z)
#'   =
#'   -\frac{1}{2}
#'   \sum_{j=1}^{d}
#'   \left(
#'     z_j^2 + \log(2\pi)
#'   \right).
#' }
#'
#' This is implemented by `log_base_prob()`.
#'
#' @section Training objective:
#' The module's `forward()` method returns a vector of log-likelihoods, one per
#' sample in the batch:
#'
#' \deqn{
#'   \ell_i = \log p_\theta(h_i \mid y_i).
#' }
#'
#' Training should minimize the negative mean log-likelihood:
#'
#' \deqn{
#'   \mathcal{L}
#'   =
#'   -\frac{1}{n}
#'   \sum_{i=1}^{n}
#'   \log p_\theta(h_i \mid y_i).
#' }
#'
#' This is not a classifier objective. The model does not estimate
#' `p(y | h)`. Instead, it estimates the embedding density given the class,
#' `p(h | y)`, or the global embedding density, `p(h)`.
#'
#' @section Embedding-quality interpretation:
#' After training, the model can assign each embedding a log-density score:
#'
#' \deqn{
#'   q_i = \log p_\theta(h_i \mid y_i).
#' }
#'
#' A high value means that the embedding is typical under its assigned class.
#' A low value means that the embedding is unusual for its assigned class. This
#' makes the model suitable for sample-quality analysis, outlier detection, and
#' out-of-distribution scoring in embedding space.
#'
#' @param embedding_dim Integer. Dimension `d` of the embedding vectors.
#' @param n_layers Integer. Number of affine coupling layers to stack.
#' @param hidden_dim Integer. Width of the hidden layers in each coupling
#'   layer's conditioner network.
#' @param n_labels Integer or `NULL`. Number of class labels. Required when
#'   `conditional = TRUE`.
#' @param conditional Logical. If `TRUE`, estimate `p(h | y)` by conditioning
#'   each coupling layer on class labels. If `FALSE`, estimate `p(h)`.
#' @param label_dim Integer. Dimension of the trainable class embedding used by
#'   conditional coupling layers.
#'
#' @return A torch module with methods:
#' \describe{
#'   \item{`log_base_prob(z)`}{Computes the standard Gaussian log-density of
#'   latent vectors.}
#'   \item{`log_prob(h, y = NULL)`}{Maps embeddings to latent space through the
#'   flow and returns log-density values.}
#'   \item{`inverse(z, y = NULL)`}{Maps latent vectors back to embedding space.}
#'   \item{`forward(input)`}{Entry point used by `luz`; returns log-density
#'   values for a batch.}
#' }
#'
#' @references
#' Dinh, L., Sohl-Dickstein, J., & Bengio, S. Density estimation using Real NVP.
#' International Conference on Learning Representations, 2017.
#'
#' @keywords internal
.flow_density_net <- torch::nn_module(
    "flow_density_net",

    initialize = function(embedding_dim,
                          n_layers = 6L,
                          hidden_dim = 128L,
                          n_labels = NULL,
                          conditional = TRUE,
                          label_dim = 16L) {

        self$embedding_dim <- embedding_dim
        self$n_layers <- n_layers
        self$conditional <- conditional

        layers <- list()

        for (k in seq_len(n_layers)) {
            mask <- rep(0, embedding_dim)

            if (k %% 2 == 1) {
                mask[seq(1, embedding_dim, by = 2)] <- 1
            } else {
                mask[seq(2, embedding_dim, by = 2)] <- 1
            }

            layers[[k]] <- .flow_coupling_layer(
                embedding_dim = embedding_dim,
                hidden_dim = hidden_dim,
                mask = mask,
                n_labels = n_labels,
                conditional = conditional,
                label_dim = label_dim
            )
        }

        self$layers <- torch::nn_module_list(layers)
    },

    log_base_prob = function(z) {
        log_2pi <- log(2 * pi)

        -0.5 * (z^2 + log_2pi)$sum(dim = 2)
    },

    log_prob = function(h, y = NULL) {
        z <- h

        total_log_det <- torch::torch_zeros(
            h$size(1),
            device = h$device
        )

        for (i in seq_len(self$n_layers)) {
            layer <- self$layers[[i]]

            out <- layer(z, y)

            z <- out$z
            total_log_det <- total_log_det + out$log_det
        }

        self$log_base_prob(z) + total_log_det
    },

    inverse = function(z, y = NULL) {
        h <- z

        total_log_det <- torch::torch_zeros(
            z$size(1),
            device = z$device
        )

        for (i in rev(seq_len(self$n_layers))) {
            layer <- self$layers[[i]]

            out <- layer$inverse(h, y)

            h <- out$x
            total_log_det <- total_log_det + out$log_det
        }

        list(
            h = h,
            log_det = total_log_det
        )
    },

    forward = function(input) {
        if (self$conditional) {
            h <- input$h
            y <- input$label

            self$log_prob(h, y)
        } else {
            self$log_prob(input, NULL)
        }
    }
)

# ---- spatial scoring helpers ----

#' @title Build a predictor data frame for a flow density model from a pixel
#'   block
#'
#' @description
#' Constructs a predictor data frame in the format expected by the
#' \code{predict_fun} closure returned by \code{\link[sits]{sits_flow_density}}.
#' The resulting data frame has three sections of columns:
#' \enumerate{
#'   \item \code{sample_id}: integer row indices (required by
#'     \code{.pred_features()}).
#'   \item \code{label}: character vector with \code{class_name} repeated for
#'     every pixel (used by the conditional flow for class conditioning).
#'   \item Feature columns: one column per embedding dimension, named according
#'     to \code{feat_names}, holding the raw embedding values read from the
#'     raster.
#' }
#'
#' Because \code{predict_fun} applies normalisation internally (using the
#' statistics stored in its closure), the embedding values supplied here should
#' be at their natural floating-point scale — i.e., already de-scaled by the
#' INT2S factor applied during \code{sits_encode}.
#'
#' @param emb_matrix Numeric matrix of shape \code{[n_pixels, embedding_dim]}.
#'   Each row is the embedding vector of one valid (non-NA) pixel.
#' @param class_name Character scalar. The class label to assign to every pixel
#'   in the \code{label} column. For a conditional flow this sets the
#'   conditioning class; for an unconditional flow it is stored but ignored by
#'   the model.
#' @param feat_names Character vector of length \code{embedding_dim}. Column
#'   names for the feature section of the data frame. These must match the names
#'   used during flow training (recoverable from
#'   \code{names(environment(flow_model)[["train_set"]])[-c(1, 2)]}).
#'
#' @return A \code{data.frame} with columns
#'   \code{c("sample_id", "label", feat_names)}, compatible with
#'   \code{.pred_features()}, \code{.pred_references()}, and
#'   \code{.pred_normalize()}.
#'
#' @keywords internal
#' @noRd
.flow_build_pred_df <- function(emb_matrix, class_name, feat_names) {
    n <- nrow(emb_matrix)
    # Start with the two mandatory metadata columns
    pred <- data.frame(
        sample_id = seq_len(n),
        label     = rep(class_name, n),
        stringsAsFactors = FALSE
    )
    # Append one column per embedding dimension
    emb_df <- as.data.frame(emb_matrix)
    colnames(emb_df) <- feat_names
    cbind(pred, emb_df)
}

#' @title Score an embeddings-cube tile with a trained flow density model
#'
#' @description
#' Processes a single tile of an \code{embeddings_cube} through a trained
#' \code{\link[sits]{sits_flow_density}} model and writes the resulting
#' per-class density scores as a \code{probs_cube} tile.
#'
#' @section Computation:
#' For each requested class \eqn{c} and each valid pixel \eqn{p}, the function
#' evaluates the (conditional) log-density:
#'
#' \deqn{
#'   q_{p,c} = \log p_\theta(h_p \mid c),
#' }
#'
#' where \eqn{h_p} is the embedding vector of pixel \eqn{p}. For an
#' unconditional model the class label is ignored and the function computes
#' \eqn{\log p_\theta(h_p)} for every band.
#'
#' @section Normalization:
#' Raw log-density values are transformed to the \code{[1, 10 000]} integer
#' range expected by \code{probs_cube} using a per-block percentile stretch.
#' Within each class band, the 2nd percentile of valid log-densities maps to 1
#' and the 98th percentile maps to 10 000. Pixels outside this range are
#' clipped. A higher stored value therefore indicates a more \emph{typical}
#' embedding for that class; a lower value indicates an outlier.
#'
#' @param emb_tile Single tile (one-row tibble) of an \code{embeddings_cube}.
#' @param label_tile Single tile (one-row tibble) of a \code{class_cube}
#'   matching \code{emb_tile} spatially.
#' @param flow_model Predict closure of class \code{"sits_flow_density"}
#'   returned by \code{\link[sits]{sits_flow_density}}.
#' @param classes Character vector of class names to score.
#' @param emb_bands Character vector of embedding band names in the tile (e.g.
#'   \code{c("E1", "E2", …)}).
#' @param feat_names Character vector of feature-column names expected by
#'   \code{flow_model} (one per embedding dimension).
#' @param block List specifying the optimal processing block (rows/cols).
#' @param roi Optional region of interest (passed to
#'   \code{.chunks_filter_spatial()}).
#' @param output_dir Character. Directory where output GeoTIFF files are
#'   written.
#' @param version Character. Version tag appended to output file names.
#' @param verbose Logical. If \code{TRUE}, print per-tile timing information.
#' @param progress Logical. If \code{TRUE}, show a progress bar during block
#'   processing.
#'
#' @return A single-tile \code{probs_cube} tibble referencing the written
#'   output file.
#'
#' @keywords internal
#' @noRd
.flow_score_tile <- function(emb_tile,
                              label_tile,
                              flow_model,
                              classes,
                              emb_bands,
                              feat_names,
                              block,
                              roi,
                              multicores,
                              output_dir,
                              version,
                              verbose,
                              progress) {
    # Build the output file path (one multiband file, one band per class)
    out_file <- .file_derived_name(
        tile       = emb_tile,
        band       = "probs",
        version    = version,
        output_dir = output_dir
    )
    # Recovery: if the output already exists and is valid, skip reprocessing
    if (file.exists(out_file)) {
        .check_recovery()
        # Rebuild labels code map (integer index → class name)
        labels_code <- stats::setNames(classes, seq_along(classes))
        density_tile <- .tile_derived_from_file(
            file          = out_file,
            band          = "probs",
            base_tile     = emb_tile,
            labels        = labels_code,
            derived_class = "probs_cube",
            update_bbox   = TRUE
        )
        return(density_tile)
    }
    # Record start time for verbose output
    tile_start_time <- .tile_classif_start(tile = emb_tile, verbose = verbose)
    # Partition the tile into processing blocks (chunks)
    chunks <- .tile_chunks_create(tile = emb_tile, overlap = 0L, block = block)
    # Optionally restrict to ROI
    update_bbox <- FALSE
    if (.has(roi)) {
        nchunks <- nrow(chunks)
        chunks <- .chunks_filter_spatial(chunks = chunks, roi = roi)
        update_bbox <- nrow(chunks) != nchunks
    }
    # Retrieve the band configuration for probs_cube storage (INT2U, scale 0.0001)
    band_conf <- .conf_derived_band(derived_class = "probs_cube", band = "probs")
    band_scale <- .scale(band_conf)   # 0.0001
    n_classes <- length(classes)
    # Process jobs in parallel - one job per chunk
    block_files <- .jobs_map_parallel_chr(chunks, function(chunk) {
        block_cur <- .block(chunk)
        block_file <- .file_block_name(
            pattern    = .file_pattern(out_file),
            block      = block_cur,
            output_dir = output_dir
        )
        # Skip already-written blocks (resume support)
        if (all(.raster_is_valid(block_file))) {
            return(block_file)
        }
        # ---- Step 1: read all embedding bands for this block ----------------
        # .tile_read_block.eo_cube applies the INT2S scale automatically,
        # returning values at their original floating-point scale.
        emb_list <- lapply(emb_bands, function(band) {
            .tile_read_block(tile = emb_tile, band = band, block = block_cur)
        })
        # Each element is a matrix [n_pixels, 1]; bind column-wise
        emb_matrix <- do.call(cbind, emb_list)  # [n_pixels, embedding_dim]
        n_pixels <- nrow(emb_matrix)
        # ---- Step 2: build NA mask (any NA in any dimension → masked) -------
        na_mask <- C_mask_na(emb_matrix)
        # Valid pixels only
        valid_emb <- emb_matrix[!na_mask, , drop = FALSE]
        n_valid <- nrow(valid_emb)
        # ---- Step 3: score each requested class at all valid pixels ---------
        # scores_mat accumulates one column per class
        scores_mat <- matrix(
            NA_real_,
            nrow     = n_pixels,
            ncol     = n_classes,
            dimnames = list(NULL, classes)
        )
        if (n_valid > 0L) {
            for (ci in seq_len(n_classes)) {
                class_name <- classes[[ci]]
                # Build a predictor data frame with the target class label.
                # The logic is inlined here (rather than calling the helper
                # .flow_build_pred_df) so that PSOCK parallel workers on
                # Windows — which only load the *installed* package — do not
                # need to resolve the helper from the sits namespace.
                n_valid_px <- nrow(valid_emb)
                pred_df <- cbind(
                    data.frame(
                        sample_id        = seq_len(n_valid_px),
                        label            = rep(class_name, n_valid_px),
                        stringsAsFactors = FALSE
                    ),
                    stats::setNames(as.data.frame(valid_emb), feat_names)
                )
                # Call the flow model: returns [n_valid, 1] log-density matrix
                log_dens <- flow_model(pred_df)   # column "log_density"
                # Place results back into the full-pixel vector (NA for masked)
                scores_mat[!na_mask, ci] <- log_dens[, 1L]
            }
        }
        # ---- Step 4: percentile-stretch each class column to [1, 10 000] ---
        # Within a block, the 2nd–98th percentile of log-densities is linearly
        # mapped to 1–10 000 so that the result fits the INT2U probs_cube
        # storage convention.  Higher values → more typical for that class.
        out_mat <- matrix(
            NA_real_,
            nrow = n_pixels,
            ncol = n_classes
        )
        for (ci in seq_len(n_classes)) {
            col <- scores_mat[, ci]
            valid_col <- col[!is.na(col)]
            if (length(valid_col) >= 2L) {
                p02 <- stats::quantile(valid_col, 0.02, names = FALSE)
                p98 <- stats::quantile(valid_col, 0.98, names = FALSE)
                # Guard against degenerate blocks (all values identical)
                if (p98 > p02) {
                    col_norm <- (col - p02) / (p98 - p02)  # → [0, 1] approx
                } else {
                    col_norm <- rep(0.5, length(col))
                    col_norm[is.na(col)] <- NA
                }
            } else {
                # Too few valid pixels in this block; fall back to mid-range
                col_norm <- rep(0.5, length(col))
                col_norm[is.na(col)] <- NA
            }
            # Clip to [0, 1] and scale to INT2U storage range [1, 10 000]
            col_norm <- pmax(0, pmin(1, col_norm))
            # Store as integer-scaled value (band_scale = 0.0001, range = 1–10000)
            out_mat[, ci] <- col_norm / band_scale
        }
        colnames(out_mat) <- classes
        # ---- Step 5: write block raster ------------------------------------
        .raster_write_block(
            files         = block_file,
            block         = block_cur,
            bbox          = .bbox(chunk),
            values        = out_mat,
            data_type     = .data_type(band_conf),
            missing_value = .miss_value(band_conf),
            crop_block    = NULL
        )
        gc()
        block_file
    }, progress = progress)

    # Rebuild the labels code map (integer index → class name)
    labels_code <- stats::setNames(classes, seq_along(classes))

    # Merge all block files into a single tile raster.
    # Pass `multicores` directly (not .jobs_multicores()) so that GDAL's
    # NUM_THREADS is set correctly even when no PSOCK cluster is running.
    density_tile <- .tile_derived_merge_blocks(
        file          = out_file,
        band          = "probs",
        labels        = labels_code,
        base_tile     = emb_tile,
        block_files   = block_files,
        derived_class = "probs_cube",
        multicores    = multicores,
        update_bbox   = update_bbox
    )
    # Clean GPU memory (if the flow model used GPU during scoring)
    .ml_gpu_clean(flow_model)
    # Print elapsed time for this tile
    .tile_classif_end(tile = emb_tile, start_time = tile_start_time,
                      verbose = verbose)
    density_tile
}
