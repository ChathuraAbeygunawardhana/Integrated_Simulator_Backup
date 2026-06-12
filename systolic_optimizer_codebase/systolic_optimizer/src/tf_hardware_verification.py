#!/usr/bin/env python3
"""
Complete TensorFlow to Hardware Verification Flow
=================================================
End-to-end flow:
1. Load TensorFlow model
2. Extract golden outputs for all layers
3. Generate test vectors (inputs + expected outputs)
4. Compare with hardware simulation/actual outputs
5. Generate detailed verification report
"""

import numpy as np
import os
import argparse
from pathlib import Path
from hardware_verification import *


class TFModelVerificationFlow:
    """
    Complete verification flow for TensorFlow models on systolic array.
    """
    
    def __init__(
        self,
        model,  # tf.keras.Model or path to model
        fp_config: FixedPointConfig = None,
        verification_config: VerificationConfig = None
    ):
        """
        Args:
            model: TensorFlow model or path to saved model
            fp_config: Fixed-point configuration
            verification_config: Verification configuration
        """
        # Try to import TensorFlow
        try:
            import tensorflow as tf
            self.tf = tf
            
            if isinstance(model, str):
                self.model = tf.keras.models.load_model(model)
            else:
                self.model = model
                
        except ImportError:
            print("Warning: TensorFlow not available. Using mock mode.")
            self.tf = None
            self.model = None
        
        self.fp_config = fp_config or FixedPointConfig()
        self.verification_config = verification_config or VerificationConfig()
        
        self.golden_extractor = GoldenModelExtractor(self.fp_config)
        self.hw_loader = HardwareOutputLoader(self.fp_config)
        self.comparator = OutputComparator(self.verification_config)
    
    def extract_and_save_golden_outputs(
        self,
        test_inputs: np.ndarray,
        output_dir: str,
        layer_names: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract golden outputs from TensorFlow model and save to files.
        
        Args:
            test_inputs: Test input data (batch, H, W, C)
            output_dir: Directory to save outputs
            layer_names: List of layer names (None = all Conv2D)
        
        Returns:
            Dictionary of golden outputs
        """
        print("\n" + "="*80)
        print("EXTRACTING GOLDEN OUTPUTS FROM TENSORFLOW MODEL")
        print("="*80)
        
        if self.model is None:
            print("Error: No TensorFlow model available")
            return {}
        
        # Extract outputs
        golden_outputs = self.golden_extractor.extract_layer_outputs(
            self.model,
            test_inputs,
            layer_names
        )
        
        # Save to files
        print(f"\nSaving golden outputs to {output_dir}")
        self.golden_extractor.save_golden_outputs(golden_outputs, output_dir)
        
        # Also save input
        input_fixed = self.fp_config.float_to_fixed(test_inputs)
        
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, 'input.npy'), test_inputs)
        np.save(os.path.join(output_dir, 'input_fixed.npy'), input_fixed)
        
        # Save input as hex
        hex_path = os.path.join(output_dir, 'input.hex')
        self.golden_extractor._save_as_hex(input_fixed, hex_path)
        
        print(f"  Saved input data")
        
        return golden_outputs
    
    def verify_from_files(
        self,
        golden_dir: str,
        hw_output_dir: str,
        layer_names: List[str]
    ) -> VerificationReport:
        """
        Verify hardware outputs against saved golden outputs.
        
        Args:
            golden_dir: Directory with golden outputs
            hw_output_dir: Directory with hardware outputs
            layer_names: List of layer names to verify
        
        Returns:
            VerificationReport
        """
        print("\n" + "="*80)
        print("VERIFYING HARDWARE OUTPUTS")
        print("="*80)
        
        report = VerificationReport()
        
        for i, layer_name in enumerate(layer_names):
            print(f"\nVerifying layer {i}: {layer_name}")
            
            # Load golden output
            golden_path = os.path.join(golden_dir, f"{layer_name}_golden.npy")
            if not os.path.exists(golden_path):
                print(f"  Warning: Golden output not found: {golden_path}")
                continue
            
            golden_output = np.load(golden_path)
            
            # Load hardware output
            # Try both .npy and .hex formats
            hw_npy_path = os.path.join(hw_output_dir, f"{layer_name}_hw.npy")
            hw_hex_path = os.path.join(hw_output_dir, f"{layer_name}_hw.hex")
            
            if os.path.exists(hw_npy_path):
                hw_output_fixed = np.load(hw_npy_path)
            elif os.path.exists(hw_hex_path):
                hw_output_fixed = self.hw_loader.load_from_hex(
                    hw_hex_path,
                    golden_output.shape
                )
            else:
                print(f"  Warning: HW output not found for {layer_name}")
                continue
            
            # Compare
            result = self.comparator.compare_outputs(
                layer_name=layer_name,
                layer_id=i,
                golden_output=golden_output,
                hw_output_fixed=hw_output_fixed
            )
            
            self.comparator.print_result(result)
            report.add_result(result)
        
        return report
    
    def generate_test_vectors(
        self,
        test_inputs: np.ndarray,
        output_dir: str,
        include_weights: bool = True
    ):
        """
        Generate complete test vectors for hardware testing.
        
        Creates:
        - Input activations (hex + npy)
        - Expected outputs per layer (hex + npy)
        - Weights per layer (hex + npy) if include_weights=True
        - Test configuration JSON
        
        Args:
            test_inputs: Test input data
            output_dir: Output directory
            include_weights: Whether to include weights
        """
        print("\n" + "="*80)
        print("GENERATING COMPLETE TEST VECTORS")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract golden outputs
        golden_outputs = self.extract_and_save_golden_outputs(
            test_inputs,
            output_dir
        )
        
        # Save weights if requested
        if include_weights and self.model is not None:
            print(f"\nExtracting weights...")
            
            for layer in self.model.layers:
                if isinstance(layer, self.tf.keras.layers.Conv2D):
                    weights, biases = layer.get_weights()
                    
                    # Convert to fixed-point
                    weights_fixed = self.fp_config.float_to_fixed(weights)
                    biases_fixed = self.fp_config.float_to_fixed(biases)
                    
                    # Save
                    layer_dir = os.path.join(output_dir, f"weights_{layer.name}")
                    os.makedirs(layer_dir, exist_ok=True)
                    
                    np.save(os.path.join(layer_dir, 'weights.npy'), weights)
                    np.save(os.path.join(layer_dir, 'weights_fixed.npy'), weights_fixed)
                    np.save(os.path.join(layer_dir, 'biases.npy'), biases)
                    np.save(os.path.join(layer_dir, 'biases_fixed.npy'), biases_fixed)
                    
                    # Save as hex
                    weights_flat = weights_fixed.flatten()
                    hex_path = os.path.join(layer_dir, 'weights.hex')
                    self.golden_extractor._save_as_hex(weights_flat.reshape(-1), hex_path)
                    
                    print(f"  Saved weights for {layer.name}")
        
        # Generate test configuration
        config = {
            'test_info': {
                'input_shape': list(test_inputs.shape),
                'num_test_samples': test_inputs.shape[0],
                'layers': list(golden_outputs.keys())
            },
            'fixed_point': {
                'total_bits': self.fp_config.total_bits,
                'frac_bits': self.fp_config.frac_bits,
                'signed': self.fp_config.signed
            },
            'verification': {
                'abs_tolerance': self.verification_config.abs_tolerance,
                'rel_tolerance': self.verification_config.rel_tolerance,
                'max_outliers_percent': self.verification_config.max_outliers_percent
            }
        }
        
        config_path = os.path.join(output_dir, 'test_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nTest configuration saved to {config_path}")
        
        print("\n" + "="*80)
        print("TEST VECTOR GENERATION COMPLETE")
        print("="*80)
        print(f"\nGenerated files in {output_dir}:")
        print(f"  • input.npy / input.hex - Test inputs")
        print(f"  • <layer>_golden.npy / .hex - Expected outputs per layer")
        if include_weights:
            print(f"  • weights_<layer>/ - Weights and biases per layer")
        print(f"  • test_config.json - Test configuration")


def demo_with_real_model():
    """Demonstrate with a simple trained model."""
    
    print("\n" + "="*90)
    print("HARDWARE VERIFICATION WITH REAL TENSORFLOW MODEL")
    print("="*90)
    
    try:
        import tensorflow as tf
        
        # Create and train a simple model
        print("\n[Step 1] Creating and training simple model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                8, (3, 3), activation='relu', padding='same',
                input_shape=(28, 28, 1), name='conv1'
            ),
            tf.keras.layers.Conv2D(
                16, (3, 3), activation='relu', padding='same', name='conv2'
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.summary()
        
        # Create test input
        test_input = np.random.randn(1, 28, 28, 1).astype(np.float32) * 0.5
        
        # Create verification flow
        flow = TFModelVerificationFlow(
            model=model,
            fp_config=FixedPointConfig(total_bits=32, frac_bits=16),
            verification_config=VerificationConfig(
                abs_tolerance=0.01,
                rel_tolerance=0.02,
                max_outliers_percent=1.0
            )
        )
        
        # Generate test vectors
        print("\n[Step 2] Generating test vectors...")
        flow.generate_test_vectors(
            test_input,
            './demo_output/test_vectors',
            include_weights=True
        )
        
        # Simulate hardware outputs (with quantization only)
        print("\n[Step 3] Simulating hardware outputs...")
        
        golden_dir = './demo_output/test_vectors'
        hw_output_dir = './demo_output/hw_outputs'
        os.makedirs(hw_output_dir, exist_ok=True)
        
        # Simulate: HW output = golden quantized (no computation error)
        layer_names = ['conv1', 'conv2']
        
        for layer_name in layer_names:
            golden = np.load(os.path.join(golden_dir, f"{layer_name}_golden.npy"))
            golden_fixed = flow.fp_config.float_to_fixed(golden)
            
            # Simulate perfect hardware (only quantization error)
            hw_output = golden_fixed
            
            np.save(os.path.join(hw_output_dir, f"{layer_name}_hw.npy"), hw_output)
        
        # Verify
        print("\n[Step 4] Running verification...")
        
        report = flow.verify_from_files(
            golden_dir,
            hw_output_dir,
            layer_names
        )
        
        # Generate report
        print("\n[Step 5] Generating report...")
        
        report.print_summary()
        report.save_to_json('./demo_output/verification_report_real.json')
        
        print("\n" + "="*90)
        print("VERIFICATION COMPLETE")
        print("="*90)
        
    except ImportError:
        print("\nTensorFlow not installed. Skipping real model demo.")
        print("Install TensorFlow to run this demo: pip install tensorflow")


def main():
    """Main entry point with CLI."""
    
    parser = argparse.ArgumentParser(
        description='Hardware verification for TensorFlow models on systolic array'
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'extract', 'verify'],
        default='demo',
        help='Operation mode'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to TensorFlow model (.h5 file)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to test input (.npy file)'
    )
    
    parser.add_argument(
        '--golden-dir',
        type=str,
        default='./golden_outputs',
        help='Directory for golden outputs'
    )
    
    parser.add_argument(
        '--hw-dir',
        type=str,
        default='./hw_outputs',
        help='Directory with hardware outputs'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./verification_report.json',
        help='Output verification report path'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Run demo
        demo_with_real_model()
    
    elif args.mode == 'extract':
        # Extract golden outputs
        if not args.model or not args.input:
            print("Error: --model and --input required for extract mode")
            return
        
        test_input = np.load(args.input)
        
        flow = TFModelVerificationFlow(model=args.model)
        flow.generate_test_vectors(
            test_input,
            args.golden_dir,
            include_weights=True
        )
    
    elif args.mode == 'verify':
        # Verify hardware outputs
        if not args.golden_dir or not args.hw_dir:
            print("Error: --golden-dir and --hw-dir required for verify mode")
            return
        
        # Load test config to get layer names
        config_path = os.path.join(args.golden_dir, 'test_config.json')
        if not os.path.exists(config_path):
            print(f"Error: Test config not found: {config_path}")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        layer_names = config['test_info']['layers']
        
        flow = TFModelVerificationFlow(model=None)
        
        report = flow.verify_from_files(
            args.golden_dir,
            args.hw_dir,
            layer_names
        )
        
        report.print_summary()
        report.save_to_json(args.output)


if __name__ == '__main__':
    main()
