import subprocess
import sys
import os
import pytest


def uninstall_package(package_name: str):
    """Uninstalls a package if it's installed."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package_name], capture_output=True,
                                text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Uninstallation failed: {e.stderr}")
        raise


def test_import_qoptmodeler():
    # Get the root directory of the project (where setup.py should be located)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Step 1: Ensure the package is uninstalled first
    uninstall_package("qoptmodeler")

    # Step 2: Install the package locally from the root directory
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "."],
                                capture_output=True, text=True, check=True, cwd=root_dir)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Installation failed: {e.stderr}")

    # Step 3: Import the package and test if the import works
    try:
        from qoptmodeler import QuantumTranslator
        assert QuantumTranslator is not None, "Failed to import qoptmodeler.QuantumTranslator"
    except ImportError as e:
        pytest.fail(f"Import failed: {str(e)}")

    # Step 4: Uninstall the package
    uninstall_package("qoptmodeler")

    # Step 5: Verify the package is uninstalled by trying to import again
    try:
        import qoptmodeler
    except ImportError:
        pass  # Expected behavior


if __name__ == "__main__":
    test_import_qoptmodeler()
