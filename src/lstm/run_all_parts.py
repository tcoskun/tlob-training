#!/usr/bin/env python3
"""
Master script to run all three parts of the depth-ratio analysis pipeline:
1. TensorFlow LSTM model training and prediction
2. Trading decision generation
3. VectorBT backtesting

This script ensures proper execution order and handles dependencies between parts.
"""

import os
import sys
import time
import subprocess
import pickle
from pathlib import Path

def check_dependencies():
    """Check if all required files exist."""
    required_files = [
        'tensorflow_part.py',
        'trading_decision_part.py', 
        'vectorbt_part.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return False
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("Error: 'data' directory not found. Please ensure your data files are in the 'data' folder.")
        return False
    
    # Check if there are CSV files in data directory
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if not csv_files:
        print("Error: No CSV files found in 'data' directory.")
        return False
    
    print(f"✓ Found {len(csv_files)} CSV files in data directory")
    return True

def run_part(part_name, script_name, description):
    """Run a specific part of the pipeline."""
    print(f"\n{'='*60}")
    print(f"RUNNING PART {part_name}: {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {part_name} completed successfully in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running {part_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("Output:")
            print(e.stdout)
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error running {part_name}: {e}")
        return False

def check_part_results(part_name, expected_files):
    """Check if a part produced the expected output files."""
    print(f"\nChecking {part_name} results...")
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ {part_name} missing expected files: {missing_files}")
        return False
    else:
        print(f"✓ {part_name} produced all expected files")
        return True

def main():
    """Main function to run the entire pipeline."""
    print("🚀 DEPTH-RATIO ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Pipeline cannot start due to missing dependencies.")
        return
    
    # Create graphs directory if it doesn't exist
    os.makedirs('graphs', exist_ok=True)
    
    # Track overall progress
    pipeline_start = time.time()
    successful_parts = 0
    total_parts = 3
    
    print(f"\n📊 Starting pipeline execution at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Part 1: TensorFlow
    if run_part("1", "tensorflow_part.py", "TensorFlow LSTM Model Training & Prediction"):
        if check_part_results("Part 1", ['models/tensorflow_results.pkl', 'models/best_lstm_model.h5']):
            successful_parts += 1
            print("✓ Part 1 results verified")
        else:
            print("❌ Part 1 results incomplete")
            return
    else:
        print("❌ Part 1 failed")
        return
    
    # Part 2: Trading Decisions
    if run_part("2", "trading_decision_part.py", "Trading Decision Generation"):
        if check_part_results("Part 2", ['trading_results.pkl']):
            successful_parts += 1
            print("✓ Part 2 results verified")
        else:
            print("❌ Part 2 results incomplete")
            return
    else:
        print("❌ Part 2 failed")
        return
    
    # Part 3: VectorBT Backtesting
    if run_part("3", "vectorbt_part.py", "VectorBT Backtesting & Analysis"):
        if check_part_results("Part 3", ['vectorbt_final_results.pkl', 'strategy_comparison.csv']):
            successful_parts += 1
            print("✓ Part 3 results verified")
        else:
            print("❌ Part 3 results incomplete")
            return
    else:
        print("❌ Part 3 failed")
        return
    
    # Pipeline completion
    pipeline_elapsed = time.time() - pipeline_start
    
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"✅ All {total_parts} parts completed successfully")
    print(f"⏱️  Total execution time: {pipeline_elapsed:.2f} seconds")
    print(f"📁 Results saved in:")
    print(f"   - tensorflow_results.pkl")
    print(f"   - trading_results.pkl") 
    print(f"   - vectorbt_final_results.pkl")
    print(f"   - strategy_comparison.csv")
    print(f"   - graphs/ (visualization plots)")
    print(f"   - best_lstm_model.h5 (trained model)")
    
    # Load and display final summary
    try:
        with open('vectorbt_final_results.pkl', 'rb') as f:
            final_results = pickle.load(f)
        
        comparison_df = final_results['comparison']
        print(f"\n📊 FINAL STRATEGY COMPARISON:")
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Find best strategy
        best_strategy = comparison_df.loc[comparison_df['Total Return [%]'].idxmax()]
        print(f"\n🏆 BEST PERFORMING STRATEGY: {best_strategy['Strategy']}")
        print(f"   Total Return: {best_strategy['Total Return [%]']:.3f}%")
        print(f"   Sharpe Ratio: {best_strategy['Sharpe Ratio']:.3f}")
        print(f"   Max Drawdown: {best_strategy['Max Drawdown [%]']:.3f}%")
        
    except Exception as e:
        print(f"\n⚠️  Could not load final results summary: {e}")
    
    print(f"\n🎯 Pipeline execution completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()