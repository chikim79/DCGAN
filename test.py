import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import zipfile
from PIL import Image
import glob
import time

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration parameters
IMAGE_SIZE = 256
BATCH_SIZE = 64
BUFFER_SIZE = 1000
LATENT_DIM = 100
EPOCHS = 150
OUTPUT_PATH = '../images'

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load and preprocess images
def load_monet_images(image_dir):
    monet_files = glob.glob(f"{image_dir}/monet_jpg/*.jpg")
    
    print(f"Found {len(monet_files)} Monet paintings")
    
    # Create dataset from image files
    monet_dataset = tf.data.Dataset.from_tensor_slices(monet_files)
    
    # Map preprocessing function to dataset
    monet_dataset = monet_dataset.map(preprocess_image)
    
    # Shuffle and batch
    monet_dataset = monet_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    
    return monet_dataset

def preprocess_image(image_path):
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize to target dimensions
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    
    # Normalize to [-1, 1] range
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    
    return image

# Build the generator model
def build_generator():
    # Weight initialization
    initializer = tf.random_normal_initializer(0., 0.02)
    
    model = keras.Sequential([
        # Input layer for noise vector
        layers.Input(shape=(LATENT_DIM,)),
        
        # First dense layer to convert noise to 3D tensor
        layers.Dense(8 * 8 * 512, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Reshape((8, 8, 512)),
        
        # Upsampling layers
        # 8x8 -> 16x16
        layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                              kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # 16x16 -> 32x32
        layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                              kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # 32x32 -> 64x64
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                              kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # 64x64 -> 128x128
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                              kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # 128x128 -> 256x256
        layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
                              kernel_initializer=initializer),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        # Output layer with tanh activation for pixel values in [-1, 1]
        layers.Conv2D(3, (5, 5), padding='same', activation='tanh', 
                     kernel_initializer=initializer),
    ])
    
    return model

# Build the discriminator model
def build_discriminator():
    # Weight initialization
    initializer = tf.random_normal_initializer(0., 0.02)
    
    model = keras.Sequential([
        # Input layer for images
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        
        # Convolutional layers with downsampling
        # 256x256 -> 128x128
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                     kernel_initializer=initializer),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        # 128x128 -> 64x64
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', 
                     kernel_initializer=initializer),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        # 64x64 -> 32x32
        layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', 
                     kernel_initializer=initializer),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        # 32x32 -> 16x16
        layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', 
                     kernel_initializer=initializer),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        # Flatten and output
        layers.Flatten(),
        layers.Dense(1)  # No activation for WGAN-like loss stability
    ])
    
    return model

# Define loss functions
# We'll use WGAN-inspired losses for better stability
def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    return real_loss + fake_loss

# Create a class for training GAN with gradient penalty
class MonetGAN(keras.Model):
    def __init__(self, generator, discriminator):
        super(MonetGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gp_weight = 10.0  # Weight for gradient penalty
        
    def compile(self, g_optimizer, d_optimizer):
        super(MonetGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
    def gradient_penalty(self, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        
        # Generate random points between real and fake images
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
            
        # Calculate gradients w.r.t. interpolated images
        gradients = tape.gradient(pred, interpolated)
        
        # Compute the gradient norm
        norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gradient_penalty
    
    # Training step WITHOUT tf.function decorator
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Generate random noise for generator input
        noise = tf.random.normal([batch_size, LATENT_DIM])
        
        # Train discriminator
        with tf.GradientTape() as d_tape:
            # Generate fake images
            fake_images = self.generator(noise, training=True)
            
            # Get discriminator predictions
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(fake_images, training=True)
            
            # Calculate losses
            d_loss = discriminator_loss(real_output, fake_output)
            
            # Add gradient penalty
            gp = self.gradient_penalty(real_images, fake_images)
            d_loss += self.gp_weight * gp
            
        # Calculate and apply discriminator gradients
        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        
        # Train generator (typically less frequently than discriminator)
        for _ in range(1):  # Adjust frequency if needed
            noise = tf.random.normal([batch_size, LATENT_DIM])
            
            with tf.GradientTape() as g_tape:
                # Generate fake images
                fake_images = self.generator(noise, training=True)
                
                # Get discriminator predictions for fake images
                fake_output = self.discriminator(fake_images, training=True)
                
                # Calculate generator loss
                g_loss = generator_loss(fake_output)
                
            # Calculate and apply generator gradients
            g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        
        return {"d_loss": d_loss, "g_loss": g_loss}

# Function to generate and save sample images during training
def generate_and_save_images(model, epoch, test_input, output_dir='samples'):
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = model(test_input, training=False)
    
    plt.figure(figsize=(10, 10))
    
    for i in range(16):
        if i < predictions.shape[0]:
            plt.subplot(4, 4, i+1)
            img = (predictions[i].numpy() + 1) * 127.5
            img = img.astype(np.uint8)
            plt.imshow(img)
            plt.axis('off')
            
    plt.savefig(f'{output_dir}/image_at_epoch_{epoch:04d}.png')
    plt.close()

# Function to generate final images
def generate_final_images(generator, num_images=7000, batch_size=100):
    print(f"Generating {num_images} Monet-style images...")
    
    total_batches = (num_images + batch_size - 1) // batch_size
    generated_count = 0
    
    for i in range(total_batches):
        current_batch_size = min(batch_size, num_images - generated_count)
        noise = tf.random.normal([current_batch_size, LATENT_DIM])
        
        generated_images = generator(noise, training=False)
        
        for j in range(current_batch_size):
            img = (generated_images[j].numpy() + 1) * 127.5
            img = img.astype(np.uint8)
            
            # Save image
            img_path = os.path.join(OUTPUT_PATH, f'monet_{generated_count:04d}.jpg')
            Image.fromarray(img).save(img_path)
            
            generated_count += 1
            
        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"Progress: {generated_count}/{num_images} images generated")
    
    print(f"Successfully generated {generated_count} images")

# Function to create submission zip file
def create_submission_zip():
    with zipfile.ZipFile('images.zip', 'w') as zipf:
        for file in os.listdir(OUTPUT_PATH):
            if file.endswith('.jpg'):
                zipf.write(os.path.join(OUTPUT_PATH, file), file)
    
    print(f"Created submission zip file with {len(os.listdir(OUTPUT_PATH))} images")

# Main function (showing manual training loop approach without custom model.fit)
def main():
    # Load Monet dataset
    print("Loading and preprocessing Monet dataset...")
    monet_dataset = load_monet_images('../input/gan-getting-started')
    
    # Create models
    print("Building generator and discriminator models...")
    generator = build_generator()
    discriminator = build_discriminator()
    
    # Create GAN model
    gan = MonetGAN(generator, discriminator)
    
    # Create optimizers
    g_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    d_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    gan.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer)
    
    # Create fixed noise for sample generation
    sample_noise = tf.random.normal([16, LATENT_DIM])
    
    # Training loop
    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Manually iterate through dataset for each epoch
        batch_count = 0
        total_d_loss = 0
        total_g_loss = 0
        
        for image_batch in monet_dataset:
            # Perform one training step
            losses = gan.train_step(image_batch)
            
            # Accumulate losses for averaging
            total_d_loss += losses['d_loss']
            total_g_loss += losses['g_loss']
            batch_count += 1
        
        # Calculate average losses
        avg_d_loss = total_d_loss / batch_count if batch_count > 0 else 0
        avg_g_loss = total_g_loss / batch_count if batch_count > 0 else 0
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS}, Time: {epoch_time:.2f}s, "
              f"D loss: {avg_d_loss:.4f}, G loss: {avg_g_loss:.4f}")
        
        # Generate and save sample images every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            generate_and_save_images(generator, epoch + 1, sample_noise)
    
    print("Training completed successfully!")
    
    # Generate final images
    print("Generating final Monet-style images...")
    generate_final_images(generator, num_images=7000)
    
    # Create submission zip
    create_submission_zip()
    
    print("Done!")

if __name__ == "__main__":
    main()