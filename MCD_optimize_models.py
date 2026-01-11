
# libraries for model building and training
import os
os.environ['TF_XLA_FLAGS'] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
import tensorflow as tf
import numpy as np
import optuna
from optuna_integration.tfkeras import TFKerasPruningCallback
from sklearn.metrics import average_precision_score
from CLR.clr_callback import CyclicLR
from MCD_build_models import CapacityCheckCallback, build_meth_model
import gc

# set tensorflow threading
tf.function(jit_compile=True)
tf.config.optimizer.set_jit(True)
tf.config.threading.set_intra_op_parallelism_threads(10)  # Parallel ops within layers
tf.config.threading.set_inter_op_parallelism_threads(10)  # Parallel independent ops

##-----------------------------------------------------------------------------------------------##

def compute_auc_pr_per_fraction(y_true, y_pred, fractions):
    """
    Compute AUC-PR for each unique spike-in fraction.
    
    Args:
        y_true (np.ndarray): True labels (one-hot encoded or binary)
        y_pred (np.ndarray): Predicted probabilities
        fractions (np.ndarray): Spike-in fractions for each sample
        
    Returns:
        dict: Dictionary mapping each unique fraction to its AUC-PR score
    """
    unique_fractions = np.unique(fractions)
    auc_pr_per_fraction = {}
    for frac in unique_fractions:
        mask = fractions == frac
        if np.sum(mask) < 2:  # Need at least 2 samples
            continue
        y_true_frac = y_true[mask]
        y_pred_frac = y_pred[mask]
        # Handle one-hot encoded labels
        if len(y_true_frac.shape) > 1 and y_true_frac.shape[1] > 1:
            # For multi-class, compute macro-averaged AUC-PR
            auc_pr = 0.0
            n_classes = y_true_frac.shape[1]
            for c in range(n_classes):
                if len(np.unique(y_true_frac[:, c])) > 1:  # Need both classes
                    auc_pr += average_precision_score(y_true_frac[:, c], y_pred_frac[:, c])
            auc_pr /= n_classes
        else:
            # Binary classification
            y_true_flat = y_true_frac.ravel() if len(y_true_frac.shape) > 1 else y_true_frac
            y_pred_flat = y_pred_frac.ravel() if len(y_pred_frac.shape) > 1 else y_pred_frac
            if len(np.unique(y_true_flat)) > 1:  # Need both classes
                auc_pr = average_precision_score(y_true_flat, y_pred_flat)
            else:
                continue
        auc_pr_per_fraction[float(frac)] = auc_pr
    return auc_pr_per_fraction

##-----------------------------------------------------------------------------------------------##

class FractionAUCPRCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to compute and log AUC-PR per spike-in fraction after each epoch.
    
    This callback computes AUC-PR for each unique spike-in fraction in both training
    and validation sets at the end of each epoch, logging them to the training history.
    
    Args:
        X_train (np.ndarray): Training features
        Y_train (np.ndarray): Training labels
        T_train (np.ndarray): Training spike-in fractions
        X_val (np.ndarray): Validation features
        Y_val (np.ndarray): Validation labels
        T_val (np.ndarray): Validation spike-in fractions
        verbose (int): Verbosity level (0=silent, 1=print per-fraction metrics)
    """
    def __init__(self, X_train, Y_train, T_train, X_val, Y_val, T_val, verbose=1):
        super().__init__()
        self.X_train = X_train
        self.Y_train = Y_train
        self.T_train = T_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.T_val = T_val
        self.verbose = verbose
        # Get unique fractions for metric names
        self.train_fractions = np.unique(T_train)
        self.val_fractions = np.unique(T_val)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Compute training AUC-PR per fraction
        y_pred_train = self.model.predict(self.X_train, verbose=0)  # type: ignore[union-attr]
        train_auc_pr = compute_auc_pr_per_fraction(self.Y_train, y_pred_train, self.T_train)
        # Compute validation AUC-PR per fraction
        y_pred_val = self.model.predict(self.X_val, verbose=0)  # type: ignore[union-attr]
        val_auc_pr = compute_auc_pr_per_fraction(self.Y_val, y_pred_val, self.T_val)
        # Log training metrics
        for frac, auc_pr in train_auc_pr.items():
            metric_name = f"auc_pr_frac_{frac:.4f}"
            logs[metric_name] = auc_pr
        # Log validation metrics
        for frac, auc_pr in val_auc_pr.items():
            metric_name = f"val_auc_pr_frac_{frac:.4f}"
            logs[metric_name] = auc_pr
        # Optionally print per-fraction AUC-PR
        if self.verbose > 0:
            print(f"\n  Epoch {epoch + 1} - AUC-PR per spike-in fraction:")
            print("    Training:")
            for frac, auc_pr in sorted(train_auc_pr.items()):
                print(f"      Frac {frac:.4f}: {auc_pr:.4f}")
            print("    Validation:")
            for frac, auc_pr in sorted(val_auc_pr.items()):
                print(f"      Frac {frac:.4f}: {auc_pr:.4f}")

##-----------------------------------------------------------------------------------------------##

def objective(trial: optuna.Trial, train_dataset, Ttrn, val_dataset,
              Tval, Xtrn_raw, Ytrn_raw, Xval_raw, Yval_raw, 
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
    5. Computes and logs AUC-PR for each spike-in fraction after each epoch
    6. Returns the maximum validation PR AUC score for maximization

    Args:
        trial (optuna.Trial): An Optuna trial object used to suggest hyperparameter values.
        train_dataset (tf.data.Dataset): Training dataset
        Ttrn (np.ndarray): Training spike-in fractions
        val_dataset (tf.data.Dataset): Validation dataset
        Tval (np.ndarray): Validation spike-in fractions
        Xtrn_raw (np.ndarray): Raw training features for per-fraction AUC-PR computation
        Ytrn_raw (np.ndarray): Raw training labels for per-fraction AUC-PR computation
        Xval_raw (np.ndarray): Raw validation features for per-fraction AUC-PR computation
        Yval_raw (np.ndarray): Raw validation labels for per-fraction AUC-PR computation
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
    proj_dim     = trial.suggest_categorical("proj_dim", [16, 32, 64, 128, 256, 512])
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
    # Create callback for per-fraction AUC-PR computation after each epoch
    fraction_auc_callback = FractionAUCPRCallback(X_train=Xtrn_raw, Y_train=Ytrn_raw, T_train=Ttrn,
                                                  X_val=Xval_raw, Y_val=Yval_raw, T_val=Tval,
                                                  verbose=1)
    callbacks = [ earlyStop, clr, pruning_callback, capacity_check, fraction_auc_callback ]
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

def study_training(Xtrn, Ytrn, Wtrn, Ttrn, Xval, Yval, Wval, Tval,
                   singleton=False, BATCH_SIZE=128) -> optuna.Study:
    """
    Conduct hyperparameter optimization study using Optuna.

    Args:
        Xtrn (np.ndarray): Training feature data
        Ytrn (np.ndarray): Training labels
        Wtrn (np.ndarray): Training sample weights
        Ttrn (np.ndarray): Training spike-in fractions
        Xval (np.ndarray): Validation feature data
        Yval (np.ndarray): Validation labels
        Wval (np.ndarray): Validation sample weights
        Tval (np.ndarray): Validation spike-in fractions
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
    # Keep raw data for per-fraction AUC-PR computation
    Xtrn_raw, Ytrn_raw = Xtrn.copy(), Ytrn.copy()
    Xval_raw, Yval_raw = Xval.copy(), Yval.copy()
    del Xtrn, Ytrn, Wtrn, Xval, Yval, Wval  # free memory
    gc.collect()
    # Create and run Optuna study
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda trial:
                   objective(trial, train_dataset, Ttrn, val_dataset,
                             Tval, Xtrn_raw, Ytrn_raw, Xval_raw, Yval_raw, 
                             singleton, BATCH_SIZE), 
                   n_trials=40, timeout=None)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    return study

##-----------------------------------------------------------------------------------------------##
