import json
import os
from typing import Any, List, Tuple
from warnings import warn

import numpy as np
from keras.activations import sigmoid
from keras.layers import Activation, BatchNormalization, Conv2DTranspose, Dense, ReLU, Reshape
from keras.models import load_model as lm, Sequential
from matplotlib.image import imread
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tqdm import tqdm


class TransposeCNN:
    def __init__(self, name: str = ''):
        if not name: name = 'TransposeCNN'
        self.name = name
        self.path = os.path.expanduser(f'~/models/BlenderCNN/')
        self.model = None
        self.load_model()
    
    def __str__(self):
        """Use the model's representation as the representation of this object."""
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        return '\n'.join(summary)
    
    def __repr__(self):
        return str(self)
    
    def __call__(self, vector: List[float]):
        """Query the model."""
        if type(vector) == list: vector = np.asarray(vector, dtype=float)
        if len(vector.shape) == 1: vector = vector.reshape((1,) + vector.shape)
        return self.model(
                vector
        )
    
    def train(
            self,
            n_epoch: int,
            features,
            values,
            batch_size: int = 32,
    ) -> None:
        """Train the model."""
        # Create useful callbacks
        cb_early_stopping = EarlyStopping(patience=5, restore_best_weights=True, verbose=2)  # Stop when val_loss stops
        cb_tensorboard = TensorBoard(log_dir='.logs')  # Measure losses during training
        cb_lr = ReduceLROnPlateau(verbose=2, patience=3)  # Reduce learning rate when val_loss stops moving
        
        # Train the model
        self.model.fit(
                epochs=n_epoch,
                x=features,
                y=values,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[cb_early_stopping, cb_tensorboard, cb_lr],
        )
        self.save_model()
    
    def save_model(self) -> None:
        """Save the current state of the model."""
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.model.save(os.path.join(self.path, f'{self.name}.h5'))
        warn("Model saved!")
    
    def load_model(self) -> None:
        """Load in the CNN model."""
        try:
            self.model = lm(os.path.join(self.path, f'{self.name}.h5'))
            warn("Model loaded successfully!")
        except OSError:
            warn("No model found, creating new one...")
            self.model = create_model(self.name)
            warn("Model initialised!")
        
        # Give an overview of the model
        self.model.summary()


def create_model(name):
    """Create the model."""
    model = Sequential(name=name)
    
    # Input
    # model.add(Reshape((1, 1, 5), input_dim=5, name='reshape'))
    
    # Initial layer
    model.add(Dense(16 * 16 * 128, input_dim=5, name='dense_input'))
    model.add(Reshape((16, 16, 128), name='reshape'))
    model.add(BatchNormalization(
            momentum=0.1,
            epsilon=1e-5,
            name=f'batch_norm_init',
    ))
    model.add(ReLU(
            name=f'ReLU_init',
    ))
    
    # Intermediate layers
    for i, filters in zip(range(1, 10), [128, 128, 128]):
        model.add(Conv2DTranspose(
                filters=filters,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                use_bias=False,  # Included in BatchNormalization
                name=f'conv2d_t_layer{i}',
        ))
        model.add(BatchNormalization(
                momentum=0.1,
                epsilon=1e-5,
                name=f'batch_norm_layer{i}',
        ))
        model.add(ReLU(
                name=f'ReLU_layer{i}',
        ))
    
    # Final layer
    model.add(Conv2DTranspose(
            filters=3,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            use_bias=True,  # Since no BatchNormalization
            name='conv2d_t_last',
    ))
    model.add(Activation(
            sigmoid,  # End with sigmoid layer to ensure values between 0..1
            name='sigmoid_last',
    ))
    
    # Compile the model
    model.compile(
            optimizer='adam',  # Adam optimiser
            loss='mse',  # Regression problem
    )
    return model


def parse_images(img_names: List[str]) -> Tuple[List[Any], List[Any]]:
    """Parse the image features and values from the images."""
    # Initialise placeholders
    features = np.zeros((len(img_names), 5), dtype=float)  # Replicated state-vectors
    values = np.zeros((len(img_names), 256, 256, 3), dtype=float)  # Image-depth is 3 (RGB)
    
    # Load in all vectors
    with open(os.path.expanduser(f'~/data/flying_dots/metadata.json'), 'r') as f:
        all_features = json.load(f)
    for i, img in enumerate(tqdm(img_names, desc='Parsing images...')):
        features[i, :] = all_features[img]
        values[i, :, :, :] = imread(os.path.expanduser(f'~/data/flying_dots/{img}.png'), format='RGB')[:, :, :3]
    return features, values
