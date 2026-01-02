"""
Fashion Recommendation System
Main entry point to run all stages

Usage:
    python main.py                    # Run all stages
    python main.py --stage 1          # Run specific stage
    python main.py --stage 1 2 3      # Run multiple stages
"""

import argparse
import sys

from config import Config
from utils import print_section


def run_stage0():
    """Run Stage 0: EDA and Baselines"""
    from stage0_eda_baselines import run_all_baselines
    return run_all_baselines()


def run_stage1():
    """Run Stage 1: Loading Dataset"""
    from stage1_load_data import run_stage1
    return run_stage1()


def run_stage2(data=None):
    """Run Stage 2: Generating Candidates"""
    from stage2_candidates import run_stage2
    return run_stage2(data)


def run_stage3(data=None):
    """Run Stage 3: Extracting Features"""
    from stage3_features import run_stage3
    return run_stage3(data)


def run_stage4a():
    """Run Stage 4A: LightGBM Training"""
    from stage4a_lightgbm import run_stage4a
    return run_stage4a()


def run_stage4b():
    """Run Stage 4B: Neural Towers Training"""
    from stage4b_neural import run_stage4b
    return run_stage4b()


def run_stage7():
    """Run Stage 7: Evaluation & Metrics"""
    from stage7_evaluation import run_stage7
    return run_stage7()


def run_all_stages():
    """Run all stages in sequence"""
    print_section("FASHION RECOMMENDATION SYSTEM")
    print("Running all stages...")
    
    # Stage 1: Load Data
    print("\n" + "=" * 80)
    print("RUNNING STAGE 1: LOADING DATA")
    print("=" * 80)
    data = run_stage1()
    
    # Stage 2: Generate Candidates
    print("\n" + "=" * 80)
    print("RUNNING STAGE 2: GENERATING CANDIDATES")
    print("=" * 80)
    candidates = run_stage2(data)
    
    # Stage 3: Extract Features
    print("\n" + "=" * 80)
    print("RUNNING STAGE 3: EXTRACTING FEATURES")
    print("=" * 80)
    data['candidates'] = candidates
    train_data, val_data = run_stage3(data)
    
    # Stage 4A: LightGBM Training
    print("\n" + "=" * 80)
    print("RUNNING STAGE 4A: LIGHTGBM TRAINING")
    print("=" * 80)
    run_stage4a()
    
    # Stage 4B: Neural Towers Training
    print("\n" + "=" * 80)
    print("RUNNING STAGE 4B: NEURAL TOWERS TRAINING")
    print("=" * 80)
    run_stage4b()
    
    # Stage 7: Evaluation
    print("\n" + "=" * 80)
    print("RUNNING STAGE 7: EVALUATION")
    print("=" * 80)
    results = run_stage7()
    
    print_section("ALL STAGES COMPLETE")
    print("Fashion Recommendation System pipeline finished successfully!")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fashion Recommendation System Pipeline'
    )
    parser.add_argument(
        '--stage', '-s',
        nargs='+',
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 7],
        help='Stages to run (0=EDA, 1=Load, 2=Candidates, 3=Features, 4=LightGBM, 5=Neural, 7=Eval)'
    )
    
    args = parser.parse_args()
    
    if args.stage is None:
        # Run all stages
        run_all_stages()
    else:
        # Run specific stages
        stage_map = {
            0: ('EDA and Baselines', run_stage0),
            1: ('Loading Dataset', run_stage1),
            2: ('Generating Candidates', run_stage2),
            3: ('Extracting Features', run_stage3),
            4: ('LightGBM Training', run_stage4a),
            5: ('Neural Towers Training', run_stage4b),
            7: ('Evaluation', run_stage7),
        }
        
        for stage in args.stage:
            if stage in stage_map:
                name, func = stage_map[stage]
                print(f"\n{'=' * 80}")
                print(f"RUNNING STAGE {stage}: {name.upper()}")
                print('=' * 80)
                func()


if __name__ == "__main__":
    main()

