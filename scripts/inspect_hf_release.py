#!/usr/bin/env python3
"""Print the immutable revision and files for a published AETV model release."""

from huggingface_hub import HfApi


info = HfApi().model_info("AETV/AETV", files_metadata=True)
print(info.sha)
for sibling in sorted(info.siblings, key=lambda item: item.rfilename):
    print(sibling.rfilename, sibling.size)
