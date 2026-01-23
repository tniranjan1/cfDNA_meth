import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras import layers # type: ignore

##-----------------------------------------------------------------------------------------------##

def binarization_layer(x: tf.Tensor) -> tf.Tensor:
    """
    Lambda layer to binarize input tensor based on random uniform thresholding.

    Args:
        x (tf.Tensor): Input tensor

    Returns:
        tf.Tensor: Binarized tensor
    """
    return K.cast(K.greater_equal(x, K.random_uniform(K.shape(x))), dtype='float16')

##-----------------------------------------------------------------------------------------------##

# Base model factory
def build_meth_model(n_cpgs, n_classes, proj_dim=128, l1_proj=1e-5, l2_proj=1e-5,
                     l2_hidden=1e-4, noise_std=0.01, dropout_proj=0.4, dropout_h1=0.3,
                     dropout_h2=0.2, use_hidden1=True, use_hidden2=False,
                     out_activation="softmax", singleton=False,
                     use_fraction_input=False) -> tf.keras.Model:
    """
    Build a methylation-based classifier model.

    Args:
        n_cpgs (int): Number of input CpG sites (optuna will not optimize this)
        n_classes (int): Number of output classes (optuna will not optimize this)
        proj_dim (int): Dimension of the projection layer (optuna hyperparameter)
        l1_proj (float): L1 regularization for projection layer (optuna hyperparameter)
        l2_proj (float): L2 regularization for projection layer (optuna hyperparameter)
        l2_hidden (float): L2 regularization for hidden layers (optuna hyperparameter)
        noise_std (float): Standard deviation for Gaussian noise layer (optuna hyperparameter)
        dropout_proj (float): Dropout rate after projection layer (optuna hyperparameter)
        dropout_h1 (float): Dropout rate after first hidden layer (if used) (optuna hyperparameter)
        dropout_h2 (float): Dropout rate after second hidden layer (if used) (optuna hyperparameter)
        use_hidden1 (bool): Whether to include the first hidden layer (optuna hyperparameter)
        use_hidden2 (bool): Whether to include a second hidden layer (optuna hyperparameter)
        out_activation (str): Activation function for output layer
        singleton (bool): Whether to use singleton mode (affects input binarization)
        use_fraction_input (bool): Whether to include spike-in fraction as auxiliary input.
            This helps the model learn fraction-specific patterns.

    Returns:
        tf.keras.Model: Compiled Keras model
    """
    inputs = layers.Input(shape=(n_cpgs,), dtype='float16', name="methylation_input")
    if singleton:
        x = layers.Lambda(lambda x: binarization_layer(x))(inputs)
    else:
        x = inputs
    x = layers.GaussianNoise(stddev=noise_std, name="input_noise")(x)
    x = layers.Dense(proj_dim, activation="linear",
        kernel_regularizer=keras.regularizers.l1_l2(l1=l1_proj, l2=l2_proj),
        name="projection_dense")(x)
    x = layers.BatchNormalization(name="projection_batchnorm")(x)
    x = layers.Activation("relu", name="projection_relu")(x)
    x = layers.Dropout(dropout_proj, name="projection_dropout")(x)
    
    # Optionally concatenate spike-in fraction as auxiliary input
    if use_fraction_input:
        fraction_input = layers.Input(shape=(1,), dtype='float32', name="fraction_input")
        # Scale fraction to similar magnitude as other features
        fraction_scaled = layers.Dense(8, activation="relu", name="fraction_embed")(fraction_input)
        x = layers.Concatenate(name="concat_fraction")([x, fraction_scaled])
    
    if use_hidden1:
        x = layers.Dense(proj_dim // 4, activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_hidden),
            name="hidden_dense_1")(x)
        x = layers.BatchNormalization(name="hidden_batchnorm_1")(x)
        x = layers.Dropout(dropout_h1, name="hidden_dropout_1")(x)
        if use_hidden2:
            x = layers.Dense(max(1, proj_dim // 4), activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_hidden),
                name="hidden_dense_2")(x)
            x = layers.Dropout(dropout_h2, name="hidden_dropout_2")(x)
    outputs = layers.Dense(n_classes, activation=out_activation,
                           dtype='float32', name="output")(x)
    
    if use_fraction_input:
        return tf.keras.models.Model(inputs=[inputs, fraction_input], outputs=outputs,  # type: ignore[possibly-unbound]
                                     name="meth_classifier_with_fraction")
    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name="meth_classifier")

##-----------------------------------------------------------------------------------------------##

class CapacityCheckCallback(tf.keras.callbacks.Callback):
    """
    Stop if model is clearly underfitting (insufficient capacity).
    """
    model: tf.keras.Model
    def __init__(self, patience=15, min_train_auc=0.65):
        super().__init__()
        self.patience = patience
        self.min_train_auc = min_train_auc
        self.epochs_below_threshold = 0
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        train_auc = logs.get('auc_pr', 0)
        val_auc = logs.get('val_auc_pr', 0)
        # Check if training AUC is stuck below minimum AND validation is much worse
        if train_auc < self.min_train_auc and (train_auc - val_auc) > 0.15:
            self.epochs_below_threshold += 1
            if self.epochs_below_threshold >= self.patience:
                print(f"Stopping: Model underfitting (train_auc={train_auc:.3f}, "
                      f"val_auc={val_auc:.3f}). Insufficient model capacity.")
                self.model.stop_training = True
        else:
            self.epochs_below_threshold = 0  # Reset counter

##-----------------------------------------------------------------------------------------------##