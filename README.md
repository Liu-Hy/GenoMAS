# GenoMAS


## What is this?

This repo has two main parts:

1. A minimal multi-agent framework inspired by Anthropic's blog on [building effective agents](https://www.anthropic.com/research/building-effective-agents). As they noted, "Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns." Following this idea, we built this framework with just enough encapsulation to make agent experiments easier. The framework provides:
   - A generic multi-agent communication protocol inspired by [MetaGPT](https://github.com/geekan/MetaGPT)
   - A Jupyter Notebook-style workflow where agents can plan, write code, execute, observe, and debug to solve tasks in multiple steps
   - Users can define custom agents with specific roles, guidelines, tools, and action units

2. An implementation of GenoMAS using this framework for the automated analysis of gene expression datasets. The input are datasets downloaded from GEO and TCGA, and the output are the identified significant genes related to a trait when considering the influence of a condition, for a list of around 1K (trait, condition) pairs.

## How to use it?

### 1. Data preparation
Download the input data from our Google Drive [folder](https://drive.google.com/drive/u/0/folders/1A25gqaIpcahle6TLJ81Qnd2VoluJ2NEe). \
You can verify data integrity with:
```bash
python validator.py --data-dir /path/to/data --validate
```

### 2. Set up environment
Create a conda environment with Python 3.10 and install the required packages:
```bash
conda create -n agent python=3.10
conda activate agent
pip install -r requirements.txt
```

### 3. Run the code
Modify `in_data_root` in `main.py` to set input data path on different devices.\
Run an experiment like this:
```bash
python main.py --version 1 --model gemini-2.0-flash-002 --api 1
```

The first time you run this, you'll get an error asking you to set up an API key, e.g. `GOOGLE_API_KEY_1` in a `.env` file. You'll need to get this API key from the LLM provider.

If you type a wrong model name, the error message will show you all the model names that work with this code.

For open-source LLMs, you can either run them locally with [Ollama Python](https://github.com/ollama/ollama-python), or use APIs (use the `--use-api` flag).

## Discussion

For questions/features/discussions, feel free to:
- Open an issue on GitHub
- Discuss in our Slack channel

## License

Internal use only. Do not distribute.