import argparse

import tensorflow as tf

from model import parse_images, TransposeCNN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_files", type=int, default=10000)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    
    # Restraint GPU usage
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # Load in datafiles
    img_names = [f"sample_{i}" for i in range(1, args.n_files + 1)]
    
    # Fetch image data (both input features as true values)
    features, values = parse_images(img_names)
    
    # Model
    model = TransposeCNN()
    
    # Train the model
    model.train(
            n_epoch=args.n_epochs,
            features=features,
            values=values,
            batch_size=args.batch_size,
    )
