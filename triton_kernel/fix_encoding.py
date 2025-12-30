import os

target_file = "BuildParentNodes.py"

# Read as raw bytes
with open(target_file, 'rb') as f:
    raw_data = f.read()

# Filter out non-ascii bytes (like \x95 or \xa0)
# We keep standard ASCII (0-127)
clean_data = bytes([b for b in raw_data if b < 128])

# Write back
with open(target_file, 'wb') as f:
    f.write(clean_data)

print(f"Cleaned {target_file}. All non-ASCII characters removed.")