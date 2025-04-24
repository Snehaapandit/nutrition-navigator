import pkg_resources

# List of packages you actually use (based on your code)
used = {'streamlit', 'tensorflow', 'numpy', 'pandas', 'plotly'}

# Get current installed packages
installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# Filter and write to requirements.txt
with open("requirements.txt", "w") as f:
    for pkg in used:
        version = installed.get(pkg)
        if version:
            f.write(f"{pkg}=={version}\n")
        else:
            print(f"⚠️ Package '{pkg}' not found in environment")
