#!/bin/bash
# 1. Clear dist to ensure only the newest version is published
echo "Cleaning dist directory..."
rm -rf dist

# 2. Build with verbose output to show included files/folders
echo "Building package verbosely..."
uv build --verbose

# 3. Print out all files and folders included in a verbose manner
echo "Included files in dist:"
ls -Rlh dist

# 4. Ensure only the newest version is in dist (redundant after rm -rf but follows instruction)
# (In a script where we didn't clear dist, we would filter here. Clearing is safer.)

# 5. Publish using the token in ../.pypi
echo "Publishing to PyPI..."
uv publish --token $(cat ../.pypi)
