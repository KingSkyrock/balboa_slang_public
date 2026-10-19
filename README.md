# balboa
UCSD CSE 167 codebase
https://cseweb.ucsd.edu/~tzli/cse167/

This is a revised version of balboa that builds on [SlangPy](https://github.com/shader-slang/slangpy). Currently only Homework 1 and 2 are completed. Future homeworks will be added soon.

# Setup
Clone the repository:
```bash
git clone https://github.com/BachiLi/balboa_slang_public.git
cd balboa_slang_public
```

We use [uv](https://docs.astral.sh/uv/) to manage Python packages and environments. Go to the linked website to install it if you haven't. Once installed, run
```bash
uv sync
```

Then try
```bash
uv run python main.py 1_1
```

You should see a window with white background.

# Platforms

Balboa depends on [SlangPy](https://github.com/shader-slang/slangpy) as a GPU programming framework, and should work on Windows, Linux, and Apple Silicon Macs. 

**Intel Macs:** SlangPy does not currently publish macOS x86_64 wheels, so setup may require building SlangPy from source. If you are using an Intel Mac and `uv sync` fails because no compatible SlangPy package is available, you may need to build SlangPy from source. Contact the staff/instructor if you run into issues.
