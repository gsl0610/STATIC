#!/bin/bash
# ============================================================
# STATIC-DSP V100 QPS 基准测试 — 远程服务器端执行脚本
# ============================================================
#
# 用法 (SSH 到 CVM 后执行):
#   cd /root/STATIC
#   bash remote_benchmark.sh              # 完整测试 (数据+QPS)
#   bash remote_benchmark.sh --qps-only   # 仅 QPS 测试 (数据已就绪)
#   bash remote_benchmark.sh --check      # 仅检查环境
#   bash remote_benchmark.sh --ab-test    # AB对比测试
#   bash remote_benchmark.sh --all        # 全流程 (数据+RQ-VAE+索引+训练+QPS)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODE="default"
for arg in "$@"; do
    case $arg in
        --qps-only) MODE="qps" ;;
        --check)    MODE="check" ;;
        --ab-test)  MODE="ab" ;;
        --all)      MODE="all" ;;
    esac
done

echo "============================================================"
echo "  STATIC-DSP V100 QPS 基准测试"
echo "  工作目录: $(pwd)"
echo "  模式: ${MODE}"
echo "============================================================"

# ─── 环境检查 ───
echo ""
echo "=== 环境检查 ==="

echo "--- OS ---"
uname -a

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "⚠ GPU 不可用"

echo ""
echo "--- Python ---"
python3 --version 2>/dev/null || { echo "ERROR: python3 不可用"; exit 1; }

echo ""
echo "--- PyTorch & CUDA ---"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'GPU Memory: {props.total_mem / 1024**3:.1f} GB')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'SM Count: {props.multi_processor_count}')
else:
    print('⚠ CUDA 不可用, 将使用 CPU')
" 2>/dev/null || {
    echo "⚠ PyTorch 未安装"
    if [ "$MODE" = "check" ]; then exit 1; fi
    echo "正在安装..."
    pip3 install --upgrade pip
    pip3 install torch>=2.1.0
    pip3 install -r requirements.txt
}

echo ""
echo "--- 依赖检查 ---"
python3 -c "
deps = ['numpy', 'pandas', 'sklearn', 'tqdm', 'yaml', 'pyarrow', 'transformers', 'peft', 'accelerate']
missing = []
for d in deps:
    try:
        __import__(d)
    except ImportError:
        missing.append(d)
if missing:
    print(f'缺少依赖: {missing}')
else:
    print('✅ 所有依赖已安装')
" 2>/dev/null

if [ "$MODE" = "check" ]; then
    echo ""
    echo "=== 环境检查完成 ==="
    exit 0
fi

# ─── 安装缺失依赖 ───
pip3 install -q numpy pandas scikit-learn tqdm PyYAML pyarrow transformers peft accelerate sentencepiece 2>/dev/null || true

# ─── 检测设备 ───
DEVICE=$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
echo ""
echo "使用设备: ${DEVICE}"

# ─── 数据准备 ───
if [ "$MODE" != "qps" ]; then
    if [ ! -f "data/processed/vocab_sizes.yaml" ]; then
        echo ""
        echo "=== 数据准备 ==="
        python3 run_pipeline.py --stage data --device ${DEVICE}
    else
        echo ""
        echo "  ✅ 数据已就绪 (data/processed/vocab_sizes.yaml)"
    fi
fi

# ─── 执行测试 ───
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="qps_benchmark_${TIMESTAMP}.log"

if [ "$MODE" = "all" ]; then
    echo ""
    echo "=== 全流程运行 ==="
    python3 run_pipeline.py --stage all --device ${DEVICE} 2>&1 | tee "${LOG_FILE}"

elif [ "$MODE" = "ab" ]; then
    echo ""
    echo "=== AB 对比测试 ==="
    python3 run_pipeline.py --stage ab_test --device ${DEVICE} 2>&1 | tee "ab_test_${TIMESTAMP}.log"

else
    echo ""
    echo "=== V100 QPS 基准测试 ==="
    python3 run_pipeline.py --stage qps_benchmark --device ${DEVICE} 2>&1 | tee "${LOG_FILE}"
fi

echo ""
echo "============================================================"
echo "  ✅ 测试完成"
echo "  日志: $(pwd)/${LOG_FILE}"
echo "============================================================"
