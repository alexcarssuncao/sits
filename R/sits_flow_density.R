

sits_flow_density <- function(embeddings = NULL,
                              conditional = TRUE,
                              epochs = 150L,
                              batch_size = 128L,
                              validation_split = 0.2,
                              optimizer = torch::optim_adamw,
                              opt_hparams = list(
                                  lr = 0.0005,
                                  eps = 1e-08,
                                  weight_decay = 7e-04
                              ),
                              lr_decay_epochs = 50L,
                              lr_decay_rate = 1.0,
                              patience = 20L,
                              min_delta = 0.01,
                              seed = NULL,
                              verbose = FALSE) {
    # set caller for error msg
    .check_set_caller("sits_flow_density")
    # Verifies if 'torch' and 'luz' packages is installed
    .check_require_packages(c("torch", "luz"))
    # documentation mode? verbose is FALSE
    verbose <- .message_verbose(verbose)
    # Function that trains a torch model based on samples
    train_fun <- function(embeddings) {
        # does not support working with DEM or other base data
        if (inherits(embeddings, "sits_base")) {
            stop(.conf("messages", "sits_train_base_data"), call. = FALSE)
        }
        # Avoid add a global variable for 'self'
        self <- NULL

        # Pre-conditions
        # TODO .check_pre_sits_flow_density()

        # Other pre-conditions:
        .check_int_parameter(seed, allow_null = TRUE)

        # Check opt_hparams
        # Get parameters list and remove the 'param' parameter
        optim_params_function <- formals(optimizer)[-1L]
        .check_opt_hparams(opt_hparams, optim_params_function)
        optim_params_function <- utils::modifyList(
            x = optim_params_function,
            val = opt_hparams
        )
        # Samples labels
        labels <- .samples_labels(embeddings)
        # Create numeric labels vector
        code_labels <- seq_along(labels)
        names(code_labels) <- labels
        # Number of labels, bands, and number of samples (used below)
        n_labels <- length(labels)
        embedding_dim <- length(.samples_bands(embeddings))
        # Data normalization
        ml_stats <- .samples_stats(embeddings)

        # Organize train and the test data
        # Data normalization
        ml_stats <- .samples_stats(embeddings)
        train_set <- .predictors(embeddings)
        train_set <- .pred_normalize(pred = train_set, stats = ml_stats)
        # Post condition: is predictor data valid?
        .check_predictors(pred = train_set, samples = embeddings)

        # Split the data into training and validation data sets
        # Create partitions different splits of the input data
        test_set <- .pred_sample(
            pred = train_set, frac = validation_split
        )
        # Remove the lines used for validation
        sel <- !train_set[["sample_id"]] %in% test_set[["sample_id"]]
        train_set <- train_set[sel, ]

        # Shuffle the data
        train_set <- train_set[sample(
            nrow(train_set), nrow(train_set)
        ), ]
        test_set <- test_set[sample(
            nrow(test_set), nrow(test_set)
        ), ]
        # number of samples
        n_samples_train <- nrow(train_set)
        n_samples_test <- nrow(test_set)

        # Organize data for model training
        train_x <- array(
            data = as.matrix(.pred_features(train_set)),
            dim = c(n_samples_train, embedding_dim)
        )
        train_y <- unname(code_labels[.pred_references(train_set)])
        # Create the test data
        test_x <- array(
            data = as.matrix(.pred_features(test_set)),
            dim = c(n_samples_test, embedding_dim)
        )
        test_y <- unname(code_labels[.pred_references(test_set)])
        # Create a torch seed (we define a new variable to allow users
        # to access this seed number from the model environment)
        torch_seed <- .torch_seed(seed)
        # Set torch seed
        torch::torch_manual_seed(torch_seed)

        train_ds <- .flow_embedding_dataset(
            x = train_x,
            y = train_y,
            conditional = conditional
        )

        valid_ds <- .flow_embedding_dataset(
            x = test_x,
            y = test_y,
            conditional = conditional
        )

        # Define the loss function for flow model
        flow_nll_loss <- function(input, target) {
            -input$mean()
        }
        # train with CPU or GPU?
        cpu_train <- .torch_cpu_train()
        # Train the model using luz
        model <- .flow_density_net
        torch_model <-
            luz::setup(
                module = model,
                loss = flow_nll_loss,
                optimizer = optimizer,
                metrics = list()
            ) |>
            luz::set_hparams(
                embedding_dim = embedding_dim,
                n_layers = 6L,
                hidden_dim = 128L,
                n_labels = n_labels,
                conditional = conditional
            ) |>
            luz::set_opt_hparams(
                !!!optim_params_function
            ) |>
            luz::fit(
                data = train_ds,
                valid_data = valid_ds,
                epochs = epochs,
                callbacks = list(
                    luz::luz_callback_early_stopping(
                        monitor = "valid_loss",
                        mode = "min",
                        patience = patience,
                        min_delta = min_delta
                    ),
                    luz::luz_callback_lr_scheduler(
                        torch::lr_step,
                        step_size = lr_decay_epochs,
                        gamma = lr_decay_rate
                    ),
                    # Gradient clipping prevents NaN loss from gradient
                    # explosions that flow models are prone to in early epochs.
                    luz::luz_callback_gradient_clip(max_norm = 1.0)
                ),
                accelerator = luz::accelerator(cpu = cpu_train),
                dataloader_options = list(
                    batch_size = batch_size,
                    shuffle = TRUE
                ),
                verbose = verbose
            )
        # Serialize trained model
        serialized_model <- force(.torch_serialize_model(torch_model$model))

        # Function that scores embedding values using the trained flow density.
        # Follows the same interface as other sits DL predict functions:
        # takes a predictor data frame, normalizes internally, and returns a
        # matrix whose single column contains per-sample log-density values.
        predict_fun <- function(values) {
            # Verifies if torch package is installed
            .check_require_packages("torch")
            # Set torch threads to 1
            suppressWarnings(torch::torch_set_num_threads(1L))
            # Unserialize model
            torch_model$model <- .torch_unserialize_model(
                model = torch_model$model,
                raw = serialized_model
            )

            n_samples <- nrow(values)

            # Normalize using the training statistics
            values <- .pred_normalize(pred = values, stats = ml_stats)

            # Build feature array [n_samples, embedding_dim]
            values_arr <- array(
                data = as.matrix(.pred_features(values)),
                dim = c(n_samples, embedding_dim)
            )

            # Build dataset — conditional flow also needs class labels
            if (conditional) {
                labels_int <- unname(code_labels[.pred_references(values)])
                score_ds <- .flow_embedding_dataset(
                    x = values_arr,
                    y = labels_int,
                    conditional = TRUE
                )
            } else {
                score_ds <- .flow_embedding_dataset(
                    x = values_arr,
                    y = NULL,
                    conditional = FALSE
                )
            }

            # GPU or CPU scoring?
            if (.torch_gpu_classification()) {
                score_batch_size <- sits_env[["batch_size"]]
                score_dl <- torch::dataloader(
                    score_ds,
                    batch_size = score_batch_size,
                    shuffle = FALSE
                )
                scores <- .try(
                    stats::predict(object = torch_model, score_dl),
                    .msg_error = .conf("messages", ".check_gpu_memory_size")
                )
            } else {
                score_dl <- torch::dataloader(
                    score_ds,
                    batch_size = n_samples,
                    shuffle = FALSE
                )
                scores <- stats::predict(object = torch_model, score_dl)
            }

            # Convert tensor to R matrix with one named column
            scores <- torch::as_array(scores)
            scores <- matrix(data = as.numeric(scores), ncol = 1)
            colnames(scores) <- "log_density"
            scores
        }

        # Set model class
        predict_fun <- .set_class(
            predict_fun,
            "torch_model",
            "sits_flow_density",
            class(predict_fun)
        )

        return(predict_fun)
    }
    # If embeddings is informed, train a model and return a predict function
    # Otherwise give back a train function to train model further
    .factory_function(embeddings, train_fun)
}
