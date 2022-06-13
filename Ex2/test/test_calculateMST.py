import pytest
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.common import *


def test_calculateMST(args=""):
    print(f"Launching with '{args}'")
    cmd = str(f"{EXECUTABLE} {args}")

    subprocess.run(cmd.split(sep=" "), check=True, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    sort_kernels = range(4)
    filter_kernels = range(3)
    partition_kernels = range(4)

    for sort_kernel in sort_kernels:
        for filter_kernel in filter_kernels:
            for partition_kernel in partition_kernels:
                test_calculateMST(
                    f"-s {sort_kernel} -m 1 -f {filter_kernel} -p {partition_kernel}")
