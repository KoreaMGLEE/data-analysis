"""
스크립트 5: Figure 1 생성

정의 A와 정의 B의 bias rate 결과를 시각화합니다.

Usage:
    python preliminary/script5_plot_figure1.py \
        --bias_rates_file ./results/merged/bias_rates.json \
        --output_file ./figures/fig1.png
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Project root 계산 (preliminary/script5_plot_figure1.py -> true_diff/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1")
    parser.add_argument("--bias_rates_file", type=str, required=True,
                        help="Bias rates JSON file from script4")
    parser.add_argument("--output_file", type=str, default="./figures/fig1.png",
                        help="Output figure file (PNG and PDF will be created)")
    parser.add_argument("--show_plot", action="store_true",
                        help="Show plot interactively")
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 5],
                        help="Figure size (width, height)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("Generate Figure 1")
    print("="*70)
    print(f"Input file: {args.bias_rates_file}")
    print(f"Output file: {args.output_file}")
    
    # Load bias rates
    print(f"\n[1/2] Loading bias rates...")
    data = load_json(args.bias_rates_file)
    bias_rates = data["bias_rates"]
    random_baseline = data["random_baseline"]["overall_bias_rate"]
    top_k_percent = data["top_k_percent"]
    
    print(f"  Random baseline bias rate: {random_baseline:.4f}")
    
    # Extract data for plotting
    k_values = []
    def_a_rates = []
    def_b_rates = []
    
    for k_percent in sorted(top_k_percent):
        k_values.append(int(k_percent))
        k_name = f"top_{int(k_percent)}"
        
        if k_name in bias_rates.get("definition_a", {}):
            def_a_rates.append(bias_rates["definition_a"][k_name]["bias_rate"])
        else:
            def_a_rates.append(None)
        
        if k_name in bias_rates.get("definition_b", {}):
            def_b_rates.append(bias_rates["definition_b"][k_name]["bias_rate"])
        else:
            def_b_rates.append(None)
    
    # Create figure with 2 panels
    print(f"\n[2/2] Creating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(args.figsize))
    
    # Panel (a): Definition A
    ax1.plot(k_values, def_a_rates, 'o-', linewidth=2, markersize=8, label='Definition A')
    ax1.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Random baseline ({random_baseline:.3f})')
    ax1.set_xlabel('Easy set size k (%)', fontsize=12)
    ax1.set_ylabel('Hypothesis-only bias rate', fontsize=12)
    ax1.set_title('(a) Definition A\n(Snapshot Loss)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xticks(k_values)
    
    # Panel (b): Definition B
    ax2.plot(k_values, def_b_rates, 's-', linewidth=2, markersize=8, 
             label='Definition B', color='orange')
    ax2.axhline(y=random_baseline, color='gray', linestyle='--', linewidth=1.5,
                label=f'Random baseline ({random_baseline:.3f})')
    ax2.set_xlabel('Easy set size k (%)', fontsize=12)
    ax2.set_ylabel('Hypothesis-only bias rate', fontsize=12)
    ax2.set_title('(b) Definition B\n(Learning Speed)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(k_values)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(args.output_file) if os.path.dirname(args.output_file) else ".", exist_ok=True)
    
    # Save PNG
    png_file = args.output_file if args.output_file.endswith('.png') else args.output_file + '.png'
    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {png_file}")
    
    # Save PDF
    pdf_file = args.output_file.replace('.png', '.pdf') if args.output_file.endswith('.png') else args.output_file + '.pdf'
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {pdf_file}")
    
    if args.show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Definition A bias rates:")
    for k, rate in zip(k_values, def_a_rates):
        if rate is not None:
            print(f"    k={k}%: {rate:.4f}")
    print(f"  Definition B bias rates:")
    for k, rate in zip(k_values, def_b_rates):
        if rate is not None:
            print(f"    k={k}%: {rate:.4f}")
    
    print("\n" + "="*70)
    print("Figure 1 generation completed!")
    print("="*70)


if __name__ == "__main__":
    main()

