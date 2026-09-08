#!/usr/bin/env python3
"""
Robust NLTK data downloader for cross-platform compatibility.
Handles Windows drive errors, missing nltk, and sets local data directory.
"""
import os
import sys
import subprocess

def ensure_nltk_installed():
    """Install nltk if not available."""
    try:
        import nltk
        return nltk
    except ImportError:
        print("NLTK not found. Installing via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk>=3.8"])
            import nltk
            print("NLTK installed successfully")
            return nltk
        except subprocess.CalledProcessError as e:
            print(f"Failed to install NLTK: {e}")
            print("Please run manually: pip install nltk>=3.8")
            sys.exit(1)

def setup_nltk_data_dir():
    """Set up a reliable NLTK data directory in user's home folder."""
    # Use home directory to avoid Windows drive errors
    # Use forward slashes to match nltk.data.path format on Windows
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data").replace("\\", "/")
    
    # Also check for environment variable override
    env_dir = os.environ.get("NLTK_DATA")
    if env_dir:
        nltk_data_dir = env_dir.replace("\\", "/")
    
    # Create the directory
    os.makedirs(nltk_data_dir, exist_ok=True)
    return nltk_data_dir

def download_nltk_resources(nltk_module, nltk_data_dir):
    """Download required NLTK resources to specified directory."""
    resources = [
        'brown',
        'treebank', 
        'gutenberg',
        'reuters',
        'universal_tagset',
        'punkt'
    ]
    
    print(f"NLTK data directory: {nltk_data_dir}")
    print("Checking/downloading required corpora...")
    
    # Set nltk.data.path to our directory FIRST (before any nltk operations)
    if nltk_data_dir not in nltk_module.data.path:
        nltk_module.data.path.insert(0, nltk_data_dir)
    
    # Also set NLTK_DATA env var for any subprocess calls
    os.environ['NLTK_DATA'] = nltk_data_dir
    
    failed = []
    for resource in resources:
        # Check multiple possible locations
        found = False
        for prefix in ['corpora/', 'tokenizers/', 'taggers/']:
            try:
                nltk_module.data.find(f'{prefix}{resource}')
                found = True
                break
            except LookupError:
                continue
        if found:
            print(f"  [OK] {resource} already available")
            continue
        
        print(f"  [DOWNLOAD] {resource}...")
        try:
            # Download with explicit download_dir to our directory
            nltk_module.download(resource, download_dir=nltk_data_dir, quiet=True, raise_on_error=True)
            
            # Extract any zip files in corpora/, tokenizers/, taggers/ subdirectories
            # NLTK downloads zips but doesn't auto-extract when download_dir is specified
            for prefix in ['corpora', 'tokenizers', 'taggers']:
                zip_path = os.path.join(nltk_data_dir, prefix, f"{resource}.zip")
                if os.path.exists(zip_path):
                    extract_dir = os.path.join(nltk_data_dir, prefix, resource)
                    print(f"  [EXTRACT] {resource} from {prefix}...")
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            zf.extractall(os.path.join(nltk_data_dir, prefix))
                        os.remove(zip_path)  # Clean up zip
                    except Exception as e:
                        print(f"  [WARNING] Failed to extract {resource} from {prefix}: {e}")
            
            # Verify extraction worked
            extracted = False
            for prefix in ['corpora/', 'tokenizers/', 'taggers/']:
                try:
                    nltk_module.data.find(f'{prefix}{resource}')
                    extracted = True
                    break
                except LookupError:
                    continue
            
            if extracted:
                print(f"  [OK] {resource} downloaded and extracted")
            else:
                print(f"  [WARNING] {resource} downloaded but not found after extraction")
                failed.append(resource)
                
        except Exception as e:
            print(f"  [ERROR] Failed to download {resource}: {e}")
            failed.append(resource)
    
    return failed

def verify_installation(nltk_module, nltk_data_dir):
    """Verify all required resources are accessible."""
    resources = ['brown', 'treebank', 'gutenberg', 'reuters', 'universal_tagset', 'punkt']
    all_ok = True
    
    print("\nVerifying installation...")
    for resource in resources:
        found = False
        for prefix in ['corpora/', 'tokenizers/', 'taggers/']:
            try:
                nltk_module.data.find(f'{prefix}{resource}')
                found = True
                break
            except LookupError:
                continue
        if found:
            print(f"  [OK] {resource} verified")
        else:
            print(f"  [MISSING] {resource} NOT FOUND")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 60)
    print("NLTK Data Downloader for NLP Assignment")
    print("=" * 60)
    
    # Step 1: Ensure nltk is installed
    nltk = ensure_nltk_installed()
    
    # Step 2: Set up data directory (avoids Windows drive errors)
    nltk_data_dir = setup_nltk_data_dir()
    
    # Step 3: Prepend our directory to nltk.data.path (takes priority)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    
    # Step 3: Download resources
    failed = download_nltk_resources(nltk, nltk_data_dir)
    
    # Step 4: Verify
    all_ok = verify_installation(nltk, nltk_data_dir)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"[WARNING] Some downloads failed: {', '.join(failed)}")
        print("You can retry by running this script again.")
        print("Or set NLTK_DATA environment variable to a valid path:")
        print(f"  export NLTK_DATA={nltk_data_dir}  (Linux/Mac)")
        print(f"  $env:NLTK_DATA = \"{nltk_data_dir}\"  (Windows PowerShell)")
        sys.exit(1)
    elif not all_ok:
        print("[WARNING] Some resources not accessible after download.")
        sys.exit(1)
    else:
        print("[OK] All NLTK data downloaded and verified successfully!")
        print(f"Data location: {nltk_data_dir}")
        print("\nYou can now run the assignment:")
        print("  cd q1_segmentation_pos && python main.py")
        print("  cd q2_dependency_parser && python parser.py")
        print("  cd q3_spelling_corrector && python spelling_corrector.py")
        print("  cd q4_integrated_editor && streamlit run app.py")

if __name__ == "__main__":
    main()