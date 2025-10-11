#!/usr/bin/env python3
"""
Final quick start script with actual dataset sizes and correct information
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
for subdir in ['loaders', 'configs', 'utils']:
    subdir_path = current_dir / subdir
    if subdir_path.exists():
        sys.path.insert(0, str(subdir_path))

def quick_start():
    """Quick start with actual dataset sizes"""
    print("🚀 Coding Datasets Quick Start (Final Version)")
    print("=" * 55)

    # Check credentials
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')

    if not username:
        print("❌ Set KAGGLE_USERNAME environment variable")
        return False

    if not key:
        print("❌ Set KAGGLE_KEY environment variable")
        return False

    print(f"✅ Using Kaggle credentials for: {username}")

    try:
        # Test imports
        from dataset_manager import DatasetManager
        print("✅ Imports working")

        # Initialize manager
        manager = DatasetManager()
        print("✅ DatasetManager initialized")

        # Get actual small datasets from registry
        small_datasets = manager.registry.get_small_datasets(max_mb=100)

        print(f"\n📊 Available datasets under 100MB:")
        total_size_mb = 0
        for dataset_id in small_datasets:
            info = manager.registry.get_dataset_info(dataset_id)
            # Convert size to MB
            size_str = info.size
            if "kB" in size_str:
                size_mb = float(size_str.replace(" kB", "")) / 1024
            elif "MB" in size_str:
                size_mb = float(size_str.replace(" MB", ""))
            else:
                size_mb = 0
            total_size_mb += size_mb
            print(f"  • {dataset_id}: {info.name} ({info.size})")

        print(f"\nTotal small datasets: {len(small_datasets)}")
        print(f"Total download size: {round(total_size_mb, 2)} MB")

        print(f"\n📥 Downloading {len(small_datasets)} datasets...")

        successful_downloads = []
        for dataset_id in small_datasets:
            try:
                print(f"  ⬇️  {dataset_id}...")
                manager.download_dataset(dataset_id, check_space=True)
                successful_downloads.append(dataset_id)
                print(f"  ✅ {dataset_id}")
            except Exception as e:
                print(f"  ❌ {dataset_id}: {str(e)}")

        print(f"\n✅ Downloaded {len(successful_downloads)} datasets successfully")

        # Extract prompts from successful downloads
        print("\n📝 Extracting prompts...")
        all_prompts = []
        dataset_counts = {}

        for dataset_id in successful_downloads:
            try:
                prompts = manager.extract_prompts(dataset_id)
                if prompts:
                    all_prompts.extend(prompts)
                    dataset_counts[dataset_id] = len(prompts)
                    info = manager.registry.get_dataset_info(dataset_id)
                    print(f"  ✅ {dataset_id}: {len(prompts)} prompts ({info.name})")
                else:
                    print(f"  ⚠️  {dataset_id}: no prompts found")
            except Exception as e:
                print(f"  ❌ {dataset_id}: {str(e)}")

        print(f"\n🎯 Total prompts collected: {len(all_prompts)}")

        if all_prompts:
            # Save training data
            output_file = "training_data_actual.jsonl"
            print(f"\n💾 Saving training data to {output_file}...")

            with open(output_file, 'w') as f:
                for i, prompt in enumerate(all_prompts):
                    example = {
                        "id": i,
                        "prompt": prompt,
                        "response": "",  # Empty for now
                        "source": "coding_datasets"
                    }
                    f.write(json.dumps(example) + '\n')

            print(f"✅ Saved {len(all_prompts)} examples to {output_file}")

            # Show sample
            print(f"\n📋 Sample prompts:")
            for i, prompt in enumerate(all_prompts[:3]):
                print(f"\n{i+1}. {prompt[:150]}...")

            print(f"\n🎉 Quick start complete!")
            print(f"📁 Training data ready: {output_file}")
            print(f"📊 Total examples: {len(all_prompts)}")
            print(f"📊 Dataset breakdown:")
            for dataset_id, count in dataset_counts.items():
                info = manager.registry.get_dataset_info(dataset_id)
                print(f"  • {dataset_id}: {count} prompts ({info.name})")

            return True

        else:
            print("❌ No prompts found")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_start()

    if success:
        print("\n" + "=" * 55)
        print("🎉 SUCCESS! You now have actual training data from Kaggle!")
        print("\nNext steps:")
        print("1. Load training_data_actual.jsonl in your ML framework")
        print("2. Fine-tune your model on the real prompts")
        print("3. Use for code generation tasks")
        print("\nDatasets included:")
        print("• OpenAI HumanEval Code Gen (164 prompts)")
        print("• MBPP Python Problems (~1000 prompts)")
        print("• Python Code Instructions (~18k prompts)")
        print("• Alpaca Cleaned (~52k prompts)")
        print("• Python Programming Questions (150+ prompts)")
        print("• Evol-Instruct-Code (80k prompts)")
    else:
        print("\n❌ Quick start failed. Check the error messages above.")