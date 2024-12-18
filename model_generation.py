import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_default_models(img_shape=(256, 256, 1), latent_dim=100):
    """
    Create default generator and discriminator models for cancer prediction
    
    Args:
        img_shape (tuple): Shape of input medical images
        latent_dim (int): Dimension of random noise vector
    
    Returns:
        tuple: (generator_model, discriminator_model)
    """
    # Generator Model
    generator = keras.Sequential([
        # Input layer
        layers.Dense(128 * 64 * 64, input_dim=latent_dim),
        layers.Reshape((64, 64, 128)),
        
        # Upsampling layers
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        # Output layer
        layers.Conv2D(1, (5, 5), padding='same', activation='tanh')
    ])

    # Discriminator Model
    discriminator = keras.Sequential([
        # Input layer
        layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', 
                      input_shape=img_shape),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        
        layers.Flatten(),
        layers.Dropout(0.4),
        
        # Binary classification output
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile discriminator
    discriminator.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return generator, discriminator

def save_default_models():
    """
    Create and save default models to disk
    """
    generator, discriminator = create_default_models()
    
    # Save models
    generator.save('generator_model.h5')
    discriminator.save('discriminator_model.h5')
    print("Default models created and saved successfully!")

# Allow direct execution to create models
if __name__ == "__main__":
    save_default_models()