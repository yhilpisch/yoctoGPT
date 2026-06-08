"""Allow running yoctoGPT as `python -m yoctoGPT`.

Delegates to the training CLI by default.
"""

from .train import main

if __name__ == "__main__":
    main()
