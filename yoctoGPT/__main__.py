"""Allow running yoctoGPT as `python -m yoctoGPT`.

(c) Dr. Yves J. Hilpisch
AI-Powered by Different LLMs.

Delegates to the training CLI by default.
"""

from .train import main

if __name__ == "__main__":
    main()
