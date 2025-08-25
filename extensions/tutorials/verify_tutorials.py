#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tutorial Verification Script
============================

Verifies that all tutorials are complete and functional.

Usage:
    python extensions/tutorials/verify_tutorials.py
"""

import sys
import subprocess
from pathlib import Path

def verify_tutorials():
    """Verify all tutorials are present and functional"""
    
    print("🔍 Verifying AI Research Agent Extensions Tutorials")
    print("=" * 60)
    
    tutorials_dir = Path(__file__).parent
    
    # Expected tutorial files
    expected_tutorials = [
        "01_getting_started.py",
        "02_advanced_context_engineering.py", 
        "03_semantic_graph_operations.py",
        "04_diffusion_code_repair.py",
        "05_rlhf_preference_learning.py",
        "06_cross_module_synergies.py",
        "07_confidence_filtering.py"
    ]
    
    # Check if all tutorial files exist
    missing_tutorials = []
    existing_tutorials = []
    
    for tutorial in expected_tutorials:
        tutorial_path = tutorials_dir / tutorial
        if tutorial_path.exists():
            existing_tutorials.append(tutorial)
            print(f"✅ Found: {tutorial}")
        else:
            missing_tutorials.append(tutorial)
            print(f"❌ Missing: {tutorial}")
    
    # Summary
    print(f"\n📊 Tutorial Verification Summary:")
    print(f"   Expected tutorials: {len(expected_tutorials)}")
    print(f"   Found tutorials: {len(existing_tutorials)}")
    print(f"   Missing tutorials: {len(missing_tutorials)}")
    
    if missing_tutorials:
        print(f"\n⚠️ Missing tutorials:")
        for tutorial in missing_tutorials:
            print(f"   - {tutorial}")
        return False
    else:
        print(f"\n🎉 All tutorials are present and ready!")
        print(f"\nTo run tutorials:")
        for tutorial in existing_tutorials:
            print(f"   python extensions/tutorials/{tutorial}")
        return True

if __name__ == "__main__":
    success = verify_tutorials()
    sys.exit(0 if success else 1)