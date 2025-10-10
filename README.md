# A Multi-Agent Framework for Scientific Discovery via Code-Driven Gene Expression Analysis

<div align="center">
  <img src="./imgs/logo.png" alt="GenoMAS Logo" width="300px"/>
  <br>
  <br>
  <a href="https://arxiv.org/abs/2507.21035">
    <img src="https://img.shields.io/badge/arXiv-2507.21035-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/topics/ai4science">
    <img src="https://img.shields.io/badge/AI4Science-blue.svg" alt="AI4Science">
  </a>
  <a href="https://github.com/topics/llm-agent">
    <img src="https://img.shields.io/badge/LLM%20Agent-orange.svg" alt="LLM Agent">
  </a>
  <a href="https://github.com/topics/multi-agent-systems">
    <img src="https://img.shields.io/badge/Multi--Agent%20Systems-green.svg" alt="Multi-Agent Systems">
  </a>
  <a href="https://github.com/topics/computational-genomics">
    <img src="https://img.shields.io/badge/Computational%20Genomics-purple.svg" alt="Computational Genomics">
  </a>
</div>

<br>

Official implementation of the paper:
>[__"GenoMAS: A Multi-Agent Framework for Scientific Discovery via Code-Driven Gene Expression Analysis"__](https://arxiv.org/abs/2507.21035)<br>
>Haoyang Liu, Yijiang Li, Haohan Wang<br>
>*UIUC, UC San Diego*

🌐 **[Website](https://liu-hy.github.io/GenoMAS/)** | 📄 **[Paper](https://arxiv.org/abs/2507.21035)** | 💻 **[Code](https://github.com/Liu-Hy/GenoMAS)**

---

## 🔬 Overview

<div align="center">
  <img src="./imgs/system_diagram.png" alt="GenoMAS System Diagram" width="85%"/>
</div>

<br>

This repository contains two main components:

### 🤖 1. Multi-Agent Framework
A minimal yet powerful framework for robust automation of scientific workflows:

- **Generic Communication Protocol**: Typed messaging mechanism for code-driven analysis
- **Notebook-Style Workflow**: Agents can plan, write code, execute, debug, and backtrack through multi-step tasks
- **Customizable Agents**: Define specific roles, guidelines, tools, and action units for your domain

**Design Philosophy:**
- Balance controllability of traditional workflows with flexibility of autonomous agents
- Provide just enough encapsulation to make agent experiments easier—simplicity matters
- Build a reliable foundation for production-level agent systems

### 🧬 2. GenoMAS Implementation
A specialized system for automated gene expression data analysis:

- **Data Sources**: Analyzes transcriptomic datasets from GEO and TCGA
- **Goal**: Identify significant genes related to traits while accounting for confounders
- **State-of-the-Art Performance**: 60.38% F1 score on the [GenoTEX benchmark](https://github.com/Liu-Hy/GenoTEX), substantially outperforming both open-domain agents and generic biomedical agents
- **Scientific Discovery**: Identifies biologically meaningful gene-trait associations, with high-confidence associations corroborated by literature and novel findings worthy of further investigation

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Usage](#-usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Environment Setup](#2-environment-setup)
  - [3. Run Experiments](#3-run-experiments)
- [Output Structure](#-output-structure)
- [Troubleshooting](#-troubleshooting)
- [Discussion](#-discussion)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📦 Usage

### 1. Data Preparation

**Download Input Data**

Our experiments use the publicly available GenoTEX benchmark. Download the input data (~42 GB) from the Google Drive [folder](https://drive.google.com/drive/folders/1kxHOyW5wNnY3Rk15xwLaM7ZZS01wGzRO), and save them under the same parent folder.

**Verify Data Integrity**
```bash
cd download
python validator.py --data-dir /path/to/data --validate
```

### 2. Environment Setup

**Create Python Environment**

Create a conda environment with Python 3.10 and install the required packages:
```bash
conda create -n genomas python=3.10
conda activate genomas
pip install -r requirements.txt
```

**Configure API Keys**

Create a `.env` file in the project root directory with at least one API key from a provider you choose. The code supports multiple providers and can use different API keys for load balancing.

Copy the template file and fill in your API keys:
```bash
cp env.example .env
# Then edit .env with your actual API keys
```

> 💡 **Tip**: For OpenAI models, the organization ID is also required. See [`env.example`](env.example) for the full template with all available configuration options.


### 3. Run Experiments

#### Understanding Key Arguments

- `--version`: Experiment version identifier (used for output file naming)
- `--model`: LLM model name (e.g., `gpt-5-mini-2025-08-07`, `claude-sonnet-4-5-20250929`, `gemini-2.5-pro`)
- `--api`: API key index to use (1, 2, 3, etc.), corresponding to `_1`, `_2`, `_3` suffixes in `.env`
- `--thinking`: Enable extended thinking mode for Claude models
- `--parallel-mode`: Parallelization strategy (`none` or `cohorts` for parallel cohort preprocessing)
- `--max-workers`: Number of concurrent workers when using `--parallel-mode cohorts`
- `--data-root`: Root directory containing input data. Defaults to `../data` (configurable in [`utils/config.py`](utils/config.py#L10))

**Role-Specific Model Configuration**: You can assign different models and/or API index to different agent roles, which will override the global `--model` and `--api`:
- `--code-reviewer-model`, `--code-reviewer-api`: Model for Code Review agent
- `--domain-expert-model`, `--domain-expert-api`: Model for Domain Expert agent
- `--data-engineer-model`, `--data-engineer-api`: Model for Data Engineer agents
- `--statistician-model`, `--statistician-api`: Model for Statistician agent
- `--planning-model`, `--planning-api`: Model for the planning mechanism

#### Example 1: Basic Run with Single Model

```bash
python main.py --version exp1 --model gpt-5-mini-2025-08-07 --api 1
```

#### Example 2: Heterogeneous LLM Configuration

The below replicates the heterogeneous configuration used in our paper, but any combination is allowed:
```bash
python main.py \
  --version exp2 \
  --model claude-sonnet-4-20250514 \
  --thinking \
  --api 1 \
  --planning-model o3-2025-04-16 \
  --planning-api 1 \
  --code-reviewer-model o3-2025-04-16 \
  --code-reviewer-api 1 \
  --domain-expert-model gemini-2.5-pro \
  --domain-expert-api 1
```

#### Example 3: Using Open-Source Models

**Local Deployment** (using the Ollama library):
```bash
# DeepSeek-R1 671B (requires substantial GPU resources)
# For higher performance, you may adapt the code to try latest SOTA open-source models
python main.py --version exp3 --model deepseek-r1:671b

# Llama 3.1 8B (suitable for testing on consumer GPUs)
python main.py --version exp4 --model llama3.1
```

**Via API** (using Novita by default):
```bash
# To use DeepSeek, we recommend third-party providers like Novita for reduced latency
python main.py --version exp5 --model deepseek-r1:671b --use-api --api 1
```

#### Example 4: Parallel Mode for Faster Execution

```bash
python main.py \
  --version exp6 \
  --model gpt-5-2025-08-07 \
  --api 1 \
  --parallel-mode cohorts \
  --max-workers 2
```

This processes up to 2 cohorts concurrently, significantly reducing wall-clock time. However, note that setting `max-workers` too large may stress the API rate limit.

#### Example 5: Generate Action Units with Human Refinement

Generate Action Unit (AU) prompts from guidelines, allowing manual editing before use:
```bash
python main.py \
  --version exp7 \
  --model claude-sonnet-4-5-20250929 \
  --thinking \
  --api 1 \
  --generate-action-units
```

This will:
1. Generate AU prompts from agent guidelines
2. Pause for manual editing of the generated files
3. Ask for confirmation before proceeding
4. Use the edited AUs for the experiment

#### Example 6: Generate Action Units in Non-Interactive Mode

Generate and use Action Units without manual intervention:
```bash
python main.py \
  --version exp8 \
  --model claude-sonnet-4-5-20250929 \
  --thinking \
  --api 1 \
  --generate-action-units \
  --non-interactive
```

This automatically generates and uses AU prompts without pausing for editing, suitable for automated pipelines.

#### 💰 Cost and Time Estimates

**Full Benchmark Run**
- **Time**: 3-5 days of continuous execution
- **Cost**: $300+ (varies by model choice and API pricing)
- **Scope**: All 1,384 (trait, condition) pairs in GenoTEX

**Small-Scale Testing**

For functionality verification without full replication:
1. Download only a few cohort datasets from GenoTEX
2. Use the `--quick-test` flag to skip statistical analysis and focus on preprocessing only:

```bash
python main.py \
  --version test_preprocess \
  --model claude-sonnet-4-5-20250929 \
  --api 1 \
  --quick-test
```

3. This allows you to evaluate **preprocessing quality** (the more challenging task for agents) without waiting for regression analysis
4. Note: Full regression analysis requires all related datasets to be preprocessed, which can be time-consuming

**⚠️ Important Notes**
- Logs are saved to `./output/log_{version}.txt`, which is a human-readable source for observing agent behaviors and diagnosing the system.
- If a model name is incorrect, the error message will list all supported models.
  - If you want to add new models, feel free to submit a pull request.

---

## 📂 Output Structure

The system generates outputs following the GenoTEX structure convention:

```
output/
├── preprocess/
│   └── {trait_name}/          # Preprocessed cohort datasets
├── regress/
│   └── {trait_name}/          # Regression analysis results
└── log_{version}.txt          # Detailed execution logs
```

> For detailed output format specifications, please refer to the [GenoTEX documentation](https://github.com/Liu-Hy/GenoTEX).

---

## 🔧 Troubleshooting

### Memory Issues

For local model deployment (Ollama), ensure sufficient GPU memory to prevent CUDA OOM issues or extended latency. Models like DeepSeek-R1 671B require multiple high-end GPUs. For testing, use smaller models like Llama 3.1 8B or Llama 3.3 70B.

> ⚠️ **Note**: Due to the complexity of genomic data, our experiments require a max input length of 20K tokens. Consider this when estimating GPU memory requirements for local deployment.

### Timeout Issues

Some models or API endpoints may experience extended latency. The system automatically adjusts timeout values based on model names, but it cannot handle all models and circumstances. If you encounter more than occasional timeout errors:

- Adjust the timeout scaler for specific models in [`utils/llm.py` (lines 240-256)](utils/llm.py#L240-L256)
- Modify the task-wise timeout argument `--max-time` accordingly

### Interruption Recovery

We have implemented checkpoint resume logic. In case an experiment run is interrupted, you can rerun the same command to continue from where it left off. Outputs of the half-finished task will be automatically cleared.

---

## 💬 Discussion

We welcome your feedback and contributions! Feel free to:

- [Open an issue](https://github.com/Liu-Hy/GenoMAS/issues) for bug reports or feature requests
- [Submit a pull request](https://github.com/Liu-Hy/GenoMAS/pulls) for improvements
- Contact the authors via email for research collaborations

---

## 📝 Citation

If you find our work useful for your research, please consider citing our paper and starring ⭐ our repository:

```bibtex
@misc{liu2025genomas,
      title={GenoMAS: A Multi-Agent Framework for Scientific Discovery via Code-Driven Gene Expression Analysis}, 
      author={Haoyang Liu and Yijiang Li and Haohan Wang},
      year={2025},
      eprint={2507.21035},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2507.21035}, 
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Haoyang Liu, Yijiang Li, Haohan Wang

---

## 🙏 Acknowledgements

We thank [Ye Zhang](https://linkedin.com/in/yezhang250), Senior Expert Data Scientist at Novartis, for his valuable suggestions based on his biomedical expertise.

---

<div align="center">
  
**Built with ❤️ by the GenoMAS Team**

[🌐 Website](https://liu-hy.github.io/GenoMAS/) • [📄 Paper](https://arxiv.org/abs/2507.21035) • [💻 GitHub](https://github.com/Liu-Hy/GenoMAS) • [🐛 Issues](https://github.com/Liu-Hy/GenoMAS/issues)

</div>

