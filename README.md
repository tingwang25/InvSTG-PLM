<h1 align="center">
Invariant Structure Learning with Pre-trained Language Models for Spatio-temporal Graph 🔥
</h1>

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data](#data)
- [Running Experiments](#running-experiments)

---

## Environment Setup

### 1. Create the conda environment

All commands below assume you are running from the repository root.

```bash
conda env create -f InvSTG-PLM/environment.yml
conda activate LLM-py3.9
```

### 2. Navigate to the source directory

All experiment scripts must be run from inside `src`:

```bash
cd InvSTG-PLM/InvSTG-PLM-main/src
```

### Platform notes

> **Linux (recommended):** This project is developed and tested on Linux. Using Linux is strongly recommended to avoid unexpected compatibility issues.

> **Windows:** The shell scripts require a Bash interpreter. You can obtain one in either of the following ways:
> - Install [Git for Windows](https://git-scm.com/download/win) (provides Git Bash), **or**
> - Add `git` to your conda environment:
>
> ```bash
> conda install -n LLM-py3.9 -c conda-forge git
> ```
>
> After installation, run the scripts from a Git Bash terminal, or from `cmd`/PowerShell after activating the conda environment that contains `git`.

---

## Data

The [PeMS08](https://dot.ca.gov/programs/traffic-operations/mpr/pems-source) dataset is used by default and loaded automatically. To evaluate on a different dataset, add it to the `data` directory.

---

## Running Experiments

Run the following commands from `InvSTG-PLM/InvSTG-PLM-main/src`.

### Standard setting

```bash
bash ../scripts/pems08.sh
```

### Out-of-Distribution (OOD) setting

```bash
bash ../scripts/pems08_ood.sh
```
