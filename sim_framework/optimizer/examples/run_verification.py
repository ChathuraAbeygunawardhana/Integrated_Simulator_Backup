#!/usr/bin/env python3
"""
Complete Verification Flow Example
==================================
This script demonstrates the complete flow from a pretrained TensorFlow model
to hardware verification on the systolic array.
"""

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
from tf_systolic_wrapper import SystolicArrayWrapper


def load_mnist_model():
    """Load or create a simple MNIST model."""
    
    # Try to load pretrained model
    model_path = 'mnist_model.h5'
    
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating and training new MNIST model...")
        
        # Load MNIST data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Preprocess
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                8, (3, 3), activation='relu', padding='same',
                input_shape=(28, 28, 1), name='conv1'
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu', padding='same', name='conv2'
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile and train
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nTraining model...")
        model.fit(
            x_train, y_train,
            epochs=3,
            batch_size=128,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        # Save model
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return model


def load_cifar10_model():
    """Load or create a simple CIFAR-10 model."""
    
    model_path = 'cifar10_model.h5'
    
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Creating and training new CIFAR-10 model...")
        
        # Load CIFAR-10
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Preprocess
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu', padding='same',
                input_shape=(32, 32, 3), name='conv1'
            ),
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu', padding='same', name='conv2'
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', padding='same', name='conv3'
            ),
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', padding='same', name='conv4'
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile and train (limited epochs for demo)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nTraining model...")
        model.fit(
            x_train[:10000], y_train[:10000],  # Use subset for demo
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\nTest accuracy: {test_acc:.4f}")
        
        # Save
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    return model


def verify_model_on_systolic(
    model_name: str = 'mnist',
    num_test_samples: int = 10,
    output_dir: str = './tb_generated'
):
    """
    Complete verification flow.
    
    Args:
        model_name: 'mnist' or 'cifar10'
        num_test_samples: Number of test images to verify
        output_dir: Output directory for test vectors
    """
    
    print("=" * 80)
    print(f"TensorFlow to Systolic Array Verification Flow")
    print(f"Model: {model_name.upper()}")
    print("=" * 80)
    
    # Load model
    if model_name.lower() == 'mnist':
        model = load_mnist_model()
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    elif model_name.lower() == 'cifar10':
        model = load_cifar10_model()
        (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_test = x_test.astype('float32') / 255.0
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)
    model.summary()
    
    # Create wrapper
    wrapper = SystolicArrayWrapper(
        array_height=4,
        array_width=4,
        data_width=32,
        frac_bits=16
    )
    
    # Verify each Conv2D layer
    conv_layers = [
        layer for layer in model.layers 
        if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    
    print("\n" + "=" * 80)
    print(f"Found {len(conv_layers)} Conv2D layers to verify")
    print("=" * 80)
    
    all_passed = True
    
    for i, layer in enumerate(conv_layers):
        print(f"\n{'=' * 80}")
        print(f"Verifying Layer {i+1}/{len(conv_layers)}: {layer.name}")
        print("=" * 80)
        
        # Test on multiple samples
        layer_passed = True
        
        for sample_idx in range(num_test_samples):
            test_input = x_test[sample_idx:sample_idx+1]
            
            print(f"\nTest sample {sample_idx + 1}/{num_test_samples}")
            print(f"  Ground truth label: {y_test[sample_idx][0]}")
            
            passed, metrics = wrapper.verify_layer(
                model, layer.name, test_input, tolerance=0.01
            )
            
            if not passed:
                layer_passed = False
                print(f"  ❌ FAILED")
            else:
                print(f"  ✓ PASSED")
        
        if layer_passed:
            print(f"\n✓ Layer {layer.name} verification PASSED on all samples")
        else:
            print(f"\n❌ Layer {layer.name} verification FAILED")
            all_passed = False
    
    # Generate complete testbench
    print("\n" + "=" * 80)
    print("Generating Hardware Test Vectors")
    print("=" * 80)
    
    test_batch = x_test[:5]  # Use first 5 samples for testbench
    wrapper.generate_testbench(model, test_batch, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    
    if all_passed:
        print("✓ ALL LAYERS PASSED SOFTWARE VERIFICATION")
        print(f"\nNext steps:")
        print(f"  1. Review test vectors in: {output_dir}")
        print(f"  2. Run hardware simulation:")
        print(f"     cd {output_dir}/layer_0_conv1")
        print(f"     make -f ../../Makefile.cocotb")
        print(f"  3. Check waveforms in simulation output")
    else:
        print("❌ SOME LAYERS FAILED VERIFICATION")
        print("\nPlease review the error messages above")
    
    print("=" * 80)
    
    return all_passed


def run_hardware_simulation(layer_dir: str):
    """
    Run hardware simulation for a specific layer.
    
    Args:
        layer_dir: Directory containing test vectors for layer
    """
    import subprocess
    
    print(f"\nRunning hardware simulation for {layer_dir}...")
    
    makefile = os.path.join(os.path.dirname(__file__), 'Makefile.cocotb')
    
    cmd = [
        'make',
        '-f', makefile,
        f'CONFIG_FILE={layer_dir}/config.json',
        f'INPUT_FILE={layer_dir}/input.hex',
        f'WEIGHT_FILE={layer_dir}/weights.hex',
        f'EXPECTED_FILE={layer_dir}/expected_output.hex'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed!")
        print(e.stdout)
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify TensorFlow models on systolic array'
    )
    parser.add_argument(
        '--model',
        choices=['mnist', 'cifar10'],
        default='mnist',
        help='Model to verify'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of test samples'
    )
    parser.add_argument(
        '--output-dir',
        default='./tb_generated',
        help='Output directory for test vectors'
    )
    parser.add_argument(
        '--run-sim',
        action='store_true',
        help='Run hardware simulation after generating test vectors'
    )
    
    args = parser.parse_args()
    
    # Run verification
    success = verify_model_on_systolic(
        args.model,
        args.samples,
        args.output_dir
    )
    
    # Optionally run hardware simulation
    if args.run_sim and success:
        print("\n" + "=" * 80)
        print("Running Hardware Simulations")
        print("=" * 80)
        
        # Find all layer directories
        layer_dirs = sorted([
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith('layer_')
        ])
        
        for layer_dir in layer_dirs:
            if not run_hardware_simulation(layer_dir):
                print(f"Hardware simulation failed for {layer_dir}")
                sys.exit(1)
        
        print("\n✓ All hardware simulations passed!")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
