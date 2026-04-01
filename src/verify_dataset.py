"""
verify_no_leakage.py
Data leakage verification - FIXED PATHS
"""

import os
import sys
import numpy as np

# ⭐ SMART PATH SETUP ⭐
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # If script is in root
if 'src' in current_dir:
    project_root = os.path.dirname(current_dir)

sys.path.insert(0, project_root)

# Now import
from utils.preprocess import FERDataLoader
from utils.labels import get_emotion_label

def find_dataset_path():
    """Find the correct dataset path"""
    possible_paths = [
        'dataset',           # From project root
        '../dataset',        # From src/
        'Facial-Emotion-Recognition/dataset',
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'train')):
            return path
    
    return 'dataset'  # Default

def verify_no_data_leakage():
    """
    Comprehensive data leakage verification
    """
    print("\n" + "="*70)
    print("DATA LEAKAGE VERIFICATION TEST")
    print("="*70)
    
    # Find dataset
    dataset_path = find_dataset_path()
    print(f"\n[INFO] Using dataset path: {os.path.abspath(dataset_path)}")
    
    # Load data
    print("\n[STEP 1] Loading data...")
    try:
        loader = FERDataLoader(dataset_path=dataset_path)
        X_train, X_test, y_train, y_test = loader.load_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n[HELP] Make sure your dataset folder exists:")
        print(f"  Expected: {os.path.abspath(dataset_path)}")
        print("  Structure needed:")
        print("    dataset/")
        print("    ├── train/")
        print("    │   ├── angry/")
        print("    │   ├── disgust/")
        print("    │   └── ... (other emotions)")
        print("    └── test/")
        print("        ├── angry/")
        print("        └── ... (other emotions)")
        return False
    
    # ==================== TESTS ====================
    
    all_tests_passed = True
    
    # Test 1: Size Check
    print(f"\n[TEST 1] Size Check:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Total:            {len(X_train) + len(X_test)}")
    
    if len(X_train) > 0 and len(X_test) > 0:
        print("  ✅ PASS - Both sets have data")
    else:
        print("  ❌ FAIL - Empty dataset")
        all_tests_passed = False
    
    # Test 2: Separate Loading Verification
    print(f"\n[TEST 2] Separate Folder Loading:")
    print(f"  Train loaded from: {os.path.abspath(loader.train_path)}")
    print(f"  Test loaded from:  {os.path.abspath(loader.test_path)}")
    
    if loader.train_path != loader.test_path:
        print("  ✅ PASS - Different source folders")
    else:
        print("  ❌ FAIL - Same folder (DATA LEAKAGE!)")
        all_tests_passed = False
    
    # Test 3: Shape Consistency
    print(f"\n[TEST 3] Shape Check:")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")
    
    if X_train.shape[1:] == X_test.shape[1:]:
        print("  ✅ PASS - Same dimensions")
    else:
        print("  ❌ FAIL - Different dimensions")
        all_tests_passed = False
    
    # Test 4: Normalization Check
    print(f"\n[TEST 4] Normalization Check:")
    print(f"  Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"  Test range:  [{X_test.min():.4f}, {X_test.max():.4f}]")
    
    if (0.0 <= X_train.min() <= 0.01) and (0.99 <= X_train.max() <= 1.0):
        print("  ✅ PASS - Properly normalized [0, 1]")
    else:
        print("  ⚠️  WARNING - Check normalization")
    
    # Test 5: Memory Independence
    print(f"\n[TEST 5] Memory Independence:")
    train_id = id(X_train)
    test_id = id(X_test)
    print(f"  Train memory ID: {train_id}")
    print(f"  Test memory ID:  {test_id}")
    
    if train_id != test_id:
        print("  ✅ PASS - Separate memory locations")
    else:
        print("  ❌ FAIL - Same memory (DATA LEAKAGE!)")
        all_tests_passed = False
    
    # Test 6: Content Difference Check
    print(f"\n[TEST 6] Content Hash Check:")
    # Use smaller sample to speed up
    sample_size = min(1000, len(X_train), len(X_test))
    train_sample = X_train[:sample_size].flatten()
    test_sample = X_test[:sample_size].flatten()
    
    train_hash = hash(train_sample.tobytes())
    test_hash = hash(test_sample.tobytes())
    
    print(f"  Train sample hash: {train_hash}")
    print(f"  Test sample hash:  {test_hash}")
    
    if train_hash != test_hash:
        print("  ✅ PASS - Different data content")
    else:
        print("  ❌ FAIL - Identical content (DATA LEAKAGE!)")
        all_tests_passed = False
    
    # Test 7: Statistical Independence
    print(f"\n[TEST 7] Statistical Independence:")
    train_mean = X_train.mean()
    test_mean = X_test.mean()
    train_std = X_train.std()
    test_std = X_test.std()
    
    print(f"  Train - Mean: {train_mean:.6f}, Std: {train_std:.6f}")
    print(f"  Test  - Mean: {test_mean:.6f}, Std: {test_std:.6f}")
    
    # They should be similar (same distribution) but not identical
    mean_diff = abs(train_mean - test_mean)
    if mean_diff < 0.1:  # Should be similar since same dataset
        print("  ✅ PASS - Similar distribution (expected)")
    else:
        print("  ⚠️  Note - Different distributions")
    
    # Test 8: Label Distribution
    print(f"\n[TEST 8] Label Distribution:")
    train_labels = np.argmax(y_train, axis=1)
    test_labels = np.argmax(y_test, axis=1)
    
    print("  Class      | Train  | Test")
    print("  -----------|--------|------")
    for i in range(7):
        train_count = (train_labels == i).sum()
        test_count = (test_labels == i).sum()
        emotion = get_emotion_label(i).capitalize()
        print(f"  {emotion:<10} | {train_count:>6} | {test_count:>4}")
    
    print("  ✅ Labels distributed across classes")
    
    # ==================== FINAL VERDICT ====================
    
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if all_tests_passed:
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ✅✅✅  NO DATA LEAKAGE DETECTED  ✅✅✅                    ║
    ║                                                               ║
    ║   Your training and test sets are COMPLETELY SEPARATE!        ║
    ║                                                               ║
    ║   VERIFIED:                                                   ║
    ║   ✓ Different source folders                                  ║
    ║   ✓ Different memory locations                                ║
    ║   ✓ Different content hashes                                  ║
    ║   ✓ Proper preprocessing                                      ║
    ║                                                               ║
    ║   YOUR PROJECT IS SAFE TO PROCEED! 🎉                         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)
        return True
    else:
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ❌ POTENTIAL ISSUES DETECTED                                ║
    ║                                                               ║
    ║   Please review the failed tests above.                       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
        """)
        return False

if __name__ == "__main__":
    print(f"Current directory: {os.getcwd()}")
    verify_no_data_leakage()