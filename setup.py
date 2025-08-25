#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Research Agent Extensions - Setup Script
==========================================

Automated setup and initialization script for the extension system.

Usage:
    python extensions/setup.py
    python extensions/setup.py --dev  # Development setup
    python extensions/setup.py --production  # Production setup
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
from pathlib import Path

class ExtensionsSetup:
    """Setup manager for AI Research Agent Extensions"""
    
    def __init__(self, mode="standard"):
        self.mode = mode
        self.extensions_dir = Path(__file__).parent
        self.root_dir = self.extensions_dir.parent
        
    def run_setup(self):
        """Run complete setup process"""
        print("🚀 AI Research Agent Extensions Setup")
        print("=" * 50)
        
        try:
            self.check_python_version()
            self.create_directories()
            self.install_dependencies()
            self.setup_configurations()
            self.initialize_data_structures()
            self.run_verification()
            
            print("\n✅ Setup completed successfully!")
            print("Next steps:")
            print("  1. Run: python extensions/tutorials/01_getting_started.py")
            print("  2. Check: extensions/README.md for documentation")
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            sys.exit(1)
    
    def check_python_version(self):
        """Check Python version compatibility"""
        print("🐍 Checking Python version...")
        
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8+ required")
        
        print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating directories...")
        
        directories = [
            "extensions/data",
            "extensions/logs", 
            "extensions/config",
            "extensions/semantic_graph",
            "extensions/preference_data",
            "extensions/prompt_templates"
        ]
        
        for directory in directories:
            dir_path = self.root_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {directory}")
    
    def install_dependencies(self):
        """Install required dependencies"""
        print("📦 Installing dependencies...")
        
        requirements_file = self.extensions_dir / "requirements.txt"
        
        if self.mode == "dev":
            # Install development dependencies
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
        else:
            # Install core dependencies only
            core_deps = [
                "torch>=1.9.0",
                "transformers>=4.20.0", 
                "networkx>=2.8",
                "jinja2>=3.1.0",
                "pyyaml>=6.0",
                "psutil>=5.9.0"
            ]
            
            for dep in core_deps:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True)
        
        print("   ✅ Dependencies installed")
    
    def setup_configurations(self):
        """Setup configuration files"""
        print("⚙️ Setting up configurations...")
        
        # Copy example configurations
        config_files = [
            ("integration_config.example.json", "integration_config.json"),
        ]
        
        for example_file, target_file in config_files:
            example_path = self.extensions_dir / "config" / example_file
            target_path = self.extensions_dir / "config" / target_file
            
            if example_path.exists() and not target_path.exists():
                shutil.copy2(example_path, target_path)
                print(f"   ✅ {target_file}")
    
    def initialize_data_structures(self):
        """Initialize data structures"""
        print("🗃️ Initializing data structures...")
        
        # Create empty data files
        data_files = [
            "extensions/semantic_graph/nodes.json",
            "extensions/semantic_graph/edges.json",
            "extensions/preference_data/preferences.jsonl"
        ]
        
        for data_file in data_files:
            file_path = self.root_dir / data_file
            
            if not file_path.exists():
                if data_file.endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump([], f)
                else:
                    file_path.touch()
                
                print(f"   ✅ {data_file}")
    
    def run_verification(self):
        """Run setup verification"""
        print("🔍 Verifying setup...")
        
        try:
            # Test imports
            sys.path.append(str(self.extensions_dir))
            
            from integration_orchestrator import AIResearchAgentExtensions
            from stage_1_observability import ObservabilityCollector
            from stage_2_context_builder import MemoryTierManager
            
            print("   ✅ Core imports successful")
            
            # Test basic initialization
            extensions = AIResearchAgentExtensions()
            print("   ✅ Extensions orchestrator created")
            
        except Exception as e:
            raise RuntimeError(f"Verification failed: {e}")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="AI Research Agent Extensions Setup")
    parser.add_argument("--dev", action="store_true", help="Development setup")
    parser.add_argument("--production", action="store_true", help="Production setup")
    
    args = parser.parse_args()
    
    if args.dev:
        mode = "dev"
    elif args.production:
        mode = "production"
    else:
        mode = "standard"
    
    setup = ExtensionsSetup(mode)
    setup.run_setup()

if __name__ == "__main__":
    main()