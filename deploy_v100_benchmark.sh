#!/bin/bash
# ============================================================
# STATIC-DSP V100 QPS 基准测试 — 一键部署到 CVM 并验证
# ============================================================
#
# CVM 实例: ins-fst9dw3n (carltestfordataai), V100 单卡 GPU
#
# 用法 (本地执行):
#   1. 设置服务器IP:
#      export CVM_HOST="<公网IP>"
#      export CVM_USER="root"          # 默认 root
#      export CVM_PASSWORD="<your_password>"  # 或使用 SSH Key
#
#   2. 一键部署+验证:
#      bash deploy_v100_benchmark.sh
#
#   3. 仅上传代码 (不运行):
#      bash deploy_v100_benchmark.sh --upload-only
#
#   4. 仅运行测试 (已上传):
#      bash deploy_v100_benchmark.sh --run-only
#
# 或者手动 SSH 到服务器后执行:
#   cd /root/STATIC && bash remote_benchmark.sh
# ============================================================

set -e

# ---------- 配置 ----------
CVM_HOST="${CVM_HOST:-<YOUR_CVM_IP>}"
CVM_USER="${CVM_USER:-root}"
CVM_PORT="${CVM_PORT:-22}"
REMOTE_DIR="/root/STATIC"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

UPLOAD_ONLY=false
RUN_ONLY=false

for arg in "$@"; do
    case $arg in
        --upload-only) UPLOAD_ONLY=true ;;
        --run-only)    RUN_ONLY=true ;;
    esac
done

echo "============================================================"
echo "  STATIC-DSP V100 QPS 基准测试部署"
echo "============================================================"
echo "  服务器: ${CVM_USER}@${CVM_HOST}:${CVM_PORT}"
echo "  远程目录: ${REMOTE_DIR}"
echo "  本地项目: ${PROJECT_DIR}"
echo "============================================================"

# ---------- 上传项目 ----------
if [ "$RUN_ONLY" = false ]; then
    echo ""
    echo "[1/3] 上传项目文件到 CVM..."
    echo ""

    # 创建远程目录
    ssh -p ${CVM_PORT} ${CVM_USER}@${CVM_HOST} "mkdir -p ${REMOTE_DIR}"

    # rsync 上传 (排除不必要的文件)
    rsync -avz --progress \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.codebuddy' \
        --exclude='data/' \
        --exclude='checkpoints/' \
        --exclude='logs/' \
        --exclude='*.egg-info' \
        --exclude='.venv' \
        --exclude='node_modules' \
        -e "ssh -p ${CVM_PORT}" \
        "${PROJECT_DIR}/" "${CVM_USER}@${CVM_HOST}:${REMOTE_DIR}/"

    echo ""
    echo "  ✅ 项目文件上传完成"
fi

if [ "$UPLOAD_ONLY" = true ]; then
    echo ""
    echo "  仅上传模式, 跳过运行。"
    echo "  SSH 到服务器执行: cd ${REMOTE_DIR} && bash remote_benchmark.sh"
    exit 0
fi

# ---------- 远程执行 ----------
echo ""
echo "[2/3] 在 CVM V100 上安装依赖..."
echo ""

ssh -p ${CVM_PORT} ${CVM_USER}@${CVM_HOST} << 'REMOTE_SETUP'
set -e
cd /root/STATIC

echo "--- 检查 GPU ---"
nvidia-smi || { echo "ERROR: nvidia-smi 不可用, 请确认 GPU 驱动已安装"; exit 1; }

echo ""
echo "--- 检查 Python & CUDA ---"
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || {
    echo "PyTorch 未安装，开始安装依赖..."
    pip3 install --upgrade pip
    pip3 install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip3 install -r requirements.txt
}

echo ""
echo "--- 确保依赖完整 ---"
pip3 install -q numpy pandas scikit-learn tqdm PyYAML pyarrow transformers peft accelerate sentencepiece 2>/dev/null

echo "  ✅ 依赖安装完成"
REMOTE_SETUP

echo ""
echo "[3/3] 运行 V100 QPS 基准测试..."
echo ""

ssh -p ${CVM_PORT} ${CVM_USER}@${CVM_HOST} << 'REMOTE_RUN'
set -e
cd /root/STATIC

echo "============================================================"
echo "  开始 V100 QPS 基准测试"
echo "============================================================"

# 先生成数据 (需要 vocab_sizes.yaml)
if [ ! -f "data/processed/vocab_sizes.yaml" ]; then
    echo "--- 生成测试数据 ---"
    python3 run_pipeline.py --stage data --device cuda
fi

echo ""
echo "--- 运行 QPS 基准测试 ---"
python3 run_pipeline.py --stage qps_benchmark --device cuda 2>&1 | tee /root/STATIC/qps_benchmark_result.log

echo ""
echo "============================================================"
echo "  V100 QPS 基准测试完成"
echo "  结果已保存: /root/STATIC/qps_benchmark_result.log"
echo "============================================================"
REMOTE_RUN

echo ""
echo "============================================================"
echo "  ✅ 全部完成!"
echo "  查看结果: ssh ${CVM_USER}@${CVM_HOST} 'cat ${REMOTE_DIR}/qps_benchmark_result.log'"
echo "============================================================"
