# Kubism

This repository contains the implementation of **Kubism: Disassembling and Reassembling K-Means Clustering for Mobile Heterogeneous Platforms** (ACM LCTES '25). Please refer to [our paper](https://doi.org/10.1145/3735452.3735537) for technical details and experimental results.

## Repository Structure

- `src/`: Core CUDA/C++ implementation (`CMakeLists.txt`, `read_input.c`, `kmcuda.h`, `kmcuda.cc`, `kmeans.cu`, `kmeans_cpu.h`, `kmeans_cpu.cc`, and auxiliary modules).
- `shell_script/`: Example dataset generation script (`make_csv.py`) and log-parsing scripts for execution time, reassignments, skip ratios, and power metrics.

## Requirements and Build

The paper evaluates Kubism on an NVIDIA Jetson AGX Orin with an Ampere GPU (`sm_87`), a 12-core Arm Cortex-A78AE CPU, and 32 GB of LPDDR5 memory. Building Kubism requires the CUDA Toolkit, CMake (3.2 or later), and GCC/G++ with OpenMP support. Python 3 is optional if you specify `-DDISABLE_PYTHON=y`. Dataset generation requires Python 3, NumPy, and pandas.

Set `-DCUDA_ARCH` to match the compute capability of your target GPU (`87` for Jetson AGX Orin).

```bash
cd src
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=87 -DDISABLE_R=y -DDISABLE_PYTHON=y .
make -j$(nproc)
gcc read_input.c -I. -L. -lKMCUDA -Wl,-rpath,. -o read_input
```

## Dataset Preparation

The `read_input` binary reads a headerless CSV file, with each row containing a data point with `D` floating-point features. The code fixes the number of clusters (`k`) at 1,024 in `src/read_input.c`. The paper evaluates nine synthetic datasets with `N` in `{500,000, 1,000,000, 2,000,000}` and `D` in `{64, 128, 256}`, where `N` denotes the number of data points and `D` denotes the number of features per data point. Each feature value follows a uniform distribution over `[0, 100]`.

To generate a dataset with uniformly distributed feature values, run the following command from `src/`.

```bash
python3 -c "import numpy as np, pandas as pd; pd.DataFrame(np.random.uniform(0, 100, (1000000, 128))).to_csv('uniform_1M_128.csv', header=False, index=False)"
```

The repository also provides `shell_script/make_csv.py` as an example script to generate normally distributed data (`rows = 100`, `columns = 128`). To use the generated data with `read_input`, specify `header=False` in `df.to_csv(...)`.

## Reproduction Guide

The `kmeans_cuda_yy` function in `src/kmeans.cu` selects an execution mode based on `flag` (line 1689 inside `if (refresh)`) and the dynamic threshold logic (lines 1996–1999 and 2207–2219). The following table describes the settings for each experimental configuration, including warp balancing (WB) and heterogeneous platform-aware task distribution (HETD).

| Configuration | `src/kmeans.cu` Setting | Log-Parsing Script |
|---|---|---|
| **Kubism** (default) | Keep `flag = 3` (line 1689) and retain the dynamic threshold switching logic (lines 1996–1999, 2207–2219) | `shell_script/extract_kubism_times.sh`, `shell_script/extract_skip_ratio.sh` |
| **Baseline** (GPU Yinyang) | Set `flag = 0` (line 1689) | `shell_script/baseline_extract_times.sh`, `shell_script/baseline_extract_reassignments.sh` |
| **WB + HETD** | Keep `flag = 3` (line 1689) and comment out dynamic switching to `flag = 4` (lines 1996–1999, 2207–2219) | `shell_script/extract_WB+HETD_times.sh` |
| **HETD Only** | Set `flag = 4` (line 1689) | `shell_script/extract_JUST_HETD_times.sh` |

After changing the settings in `src/kmeans.cu`, rebuild `libKMCUDA.so` by running `make -j$(nproc)` in `src/`. Run `./read_input dataset.csv > output.log` with your dataset and parse the output using the corresponding script in `shell_script/`.

Example execution and log parsing (from `src/`)

```bash
./read_input uniform_1M_128.csv > kubism.log
bash ../shell_script/extract_kubism_times.sh kubism.log
bash ../shell_script/extract_skip_ratio.sh kubism.log
```

The `kubism_CPU_GPU_task_partition` kernel (`src/kmeans.cu`, lines 847–974) uses sample-index thresholds tuned for the Jetson AGX Orin evaluation datasets.

### Optional Power Measurement with `tegrastats`

To measure CPU and GPU power on NVIDIA Jetson platforms, uncomment `capture_power_metrics_in_interval_start(iter)` (line 1888) and `capture_power_metrics_in_interval_end(iter)` (line 2402) in `src/kmeans.cu`. Update the log output paths on lines 1172 and 1187 and rebuild `libKMCUDA.so`. Run `read_input` with `sudo` to collect power measurements using `tegrastats`.

Use the following scripts from the repository root to parse the generated `tegra_log_*.txt` files. Replace `<tegra_log_dir>` with the directory containing the log files.

```bash
bash shell_script/extract_power_CPU_metrics.sh <tegra_log_dir>
bash shell_script/extract_power_GPU_metrics.sh <tegra_log_dir>
```

## Citation

If you use Kubism in your research, please cite our LCTES '25 paper.

```bibtex
@inproceedings{kim2025kubism,
  author    = {Kim, Seondeok and Choi, Sangun and Jeon, Jaebeom and Kim, Junsu and Gil, Minseong and Ryu, Jaehyeok and Oh, Yunho},
  title     = {Kubism: Disassembling and Reassembling K-Means Clustering for Mobile Heterogeneous Platforms},
  booktitle = {Proceedings of the 26th ACM SIGPLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems (LCTES '25)},
  year      = {2025},
  pages     = {1--12},
  publisher = {ACM},
  doi       = {10.1145/3735452.3735537}
}
```