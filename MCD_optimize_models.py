
# libraries for model building and training
import os
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
import tensorflow as tf
import numpy as np
import optuna
from optuna_integration.tfkeras import TFKerasPruningCallback
from CLR.clr_callback import CyclicLR
from MCD_build_models import CapacityCheckCallback, build_meth_model
import gc

# set tensorflow threading
tf.function(jit_compile=True)
tf.config.optimizer.set_jit(True)
tf.config.threading.set_intra_op_parallelism_threads(10)  # Parallel ops within layers
tf.config.threading.set_inter_op_parallelism_threads(10)  # Parallel independent ops

##-----------------------------------------------------------------------------------------------##

def objective(trial: optuna.Trial, train_dataset, val_dataset,
              singleton=False, BATCH_SIZE=128) -> float:
    """
    Objective function for Optuna hyperparameter optimization of a methylation classification model.

    This function defines the search space for hyperparameter tuning and trains a neural network
    model with the suggested hyperparameters. It evaluates model performance on a validation set
    and returns the validation PR AUC score (or validation loss as fallback) for optimization.

    The function performs the following steps:
    1. Sets random seeds for reproducibility across trials
    2. Suggests hyperparameters from predefined search spaces:
        - proj_dim: projection layer dimension (categorical values: 43, 64, 128, 256, 512)
        - l1_proj, l2_proj, l2_hidden: L1/L2 regularization coefficients (log-uniform ranges)
        - noise_std: standard deviation of Gaussian noise layer (log-uniform range)
        - dropout_proj, dropout_h1: dropout rates for projection and hidden layers
        - use_hidden2: whether to include a second hidden layer
        - lr: learning rate for Adam optimizer (log-uniform range)
        - label_smooth: label smoothing coefficient for loss function
    3. Builds and compiles a methylation classification model with suggested hyperparameters
    4. Trains the model with early stopping and learning rate reduction callbacks
    5. Returns the maximum validation PR AUC score for maximization

    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameter values.
        train_dataset (tf.data.Dataset): Training dataset
        val_dataset (tf.data.Dataset): Validation dataset
        singleton (bool): Whether to use singleton mode (affects data augmentation)
        BATCH_SIZE (int): Batch size for training

    Returns:
        float: The validation PR AUC score (or negative validation loss as fallback) to be
        maximized.

    Note:
         Global variables Xtr, Ytr, Xval, Yval, Wtr, Wval are expected to be in scope.
         The build_meth_model function must be defined and available.
    """
    # Fix seeds to reduce trial variance (Lambda layer remains stochastic)
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    # Search space
    proj_dim     = trial.suggest_categorical("proj_dim", [32, 64, 128, 256, 512, 1024])
    l1_proj      = trial.suggest_float("l1_proj", 1e-6, 1e-2, log=True)
    l2_proj      = trial.suggest_float("l2_proj", 1e-6, 1e-2, log=True)
    l2_hidden    = trial.suggest_float("l2_hidden", 1e-6, 1e-2, log=True)
    noise_std    = trial.suggest_float("noise_std", 1e-3, 5e-2, log=True)
    dropout_proj = trial.suggest_float("dropout_proj", 0.1, 0.6)
    dropout_h1   = trial.suggest_float("dropout_h1", 0.1, 0.6)
    dropout_h2   = trial.suggest_float("dropout_h2", 0.1, 0.6)
    use_hidden1  = trial.suggest_categorical("use_hidden1", [False, True])
    use_hidden2  = trial.suggest_categorical("use_hidden2", [False, True])
    start_lr     = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.05)
    # Define loss and final activation functions
    n_cpgs = train_dataset.element_spec[0].shape[1]
    n_classes = train_dataset.element_spec[1].shape[1]
    if n_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        out_activation = 'sigmoid'
    else:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smooth)
        out_activation = 'softmax'
    # Build model
    model = build_meth_model(n_cpgs=n_cpgs, n_classes=n_classes, proj_dim=proj_dim,
                             l1_proj=l1_proj, l2_proj=l2_proj, l2_hidden=l2_hidden,
                             noise_std=noise_std, dropout_proj=dropout_proj,
                             dropout_h1=dropout_h1, dropout_h2=dropout_h2,
                             use_hidden1=use_hidden1, use_hidden2=use_hidden2,
                             out_activation=out_activation, singleton=singleton)
    metrics = [ tf.keras.metrics.AUC(multi_label=False, name="auc_roc"),
                tf.keras.metrics.AUC(curve="PR", multi_label=False, name="auc_pr"),
                tf.keras.metrics.Precision(name="precision"), 
                tf.keras.metrics.Recall(name="recall") ]
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=start_lr)
    clr = CyclicLR(base_lr=1e-7, max_lr=start_lr, step_size=2*steps_per_epoch,
                   mode='triangular', monitor='val_loss', patience=8, factor=0.5, min_delta=0.99,
                   min_max_lr=1e-7, verbose=1)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12)
    pruning_callback = TFKerasPruningCallback(trial, "val_auc_pr")
    capacity_check = CapacityCheckCallback(patience=15, min_train_auc=0.65)
    model.compile(optimizer=optimizer, loss=loss, weighted_metrics = metrics, jit_compile=True)
    callbacks = [ earlyStop, clr, pruning_callback, capacity_check ]
    # print summary of hyperparameters for this trial
    print(f"Trial {trial.number} hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    # Train model
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=1000,
                        batch_size=BATCH_SIZE,
                        verbose=2,  # type: ignore[arg-type]
                        shuffle=True,
                        callbacks=callbacks,
                        use_multiprocessing=True,
                        workers=10,
                        max_queue_size=10)
    # Optimize for validation PR AUC if available; fallback to ROC AUC and then val_loss
    if history is None:
        return 0.0
    hist = history.history
    if "val_auc_pr" in hist:
        return max(hist["val_auc_pr"])
    if "val_auc_roc" in hist:  # name varies; prefer PR AUC
        score = max(hist["val_auc_roc"])
        return score
    return -min(hist["val_loss"])  # if no PR AUC, minimize loss

##-----------------------------------------------------------------------------------------------##

def study_training(Xtrn, Ytrn, Wtrn, Xval, Yval, Wval,
                   singleton=False, BATCH_SIZE=128) -> optuna.Study:
    """
    Conduct hyperparameter optimization study using Optuna.

    Args:
        training_data (tf.data.Dataset): Training dataset
        validation_data (tf.data.Dataset): Validation dataset
        singleton (bool): Whether to use singleton mode (affects data augmentation)
        BATCH_SIZE (int): Batch size for training

    Returns:
        optuna.Study: The completed Optuna study object containing optimization results.
    """
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((Xtrn, Ytrn, Wtrn))
    train_dataset = train_dataset.cache()  # Cache in memory
    train_dataset = train_dataset.shuffle(buffer_size=Xtrn.shape[0], reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(AUTOTUNE)  # Prefetch batches
    val_dataset = tf.data.Dataset.from_tensor_slices((Xval, Yval, Wval))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.prefetch(AUTOTUNE)
    del Xtrn, Ytrn, Wtrn, Xval, Yval, Wval  # free memory
    gc.collect()
    # Create and run Optuna study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial:
                   objective(trial, train_dataset, val_dataset, singleton, BATCH_SIZE), 
                   n_trials=40, timeout=None)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    return study

##-----------------------------------------------------------------------------------------------##
