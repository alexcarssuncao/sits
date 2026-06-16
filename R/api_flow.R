
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
            self$label_emb <- torch::nn_embedding(
                num_embeddings = n_labels,
                embedding_dim = label_dim
            )
            cond_dim <- cond_dim + label_dim
        }

        self$net <- torch::nn_sequential(
            torch::nn_linear(cond_dim, hidden_dim),
            torch::nn_relu(),
            torch::nn_linear(hidden_dim, hidden_dim),
            torch::nn_relu(),
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
