#!/usr/bin/env bash
# =============================================================================
# Music Data Process Pipeline — 一键环境安装
#
# 用法:
#   bash setup.sh              # 安装全部依赖 + 下载权重
#   bash setup.sh --weights    # 仅下载 SongFormer 权重
#   bash setup.sh --deps       # 仅安装 Python 依赖
#
# 注意:
#   - SongFormer 权重约 8 GB，默认从 HuggingFace 下载
#   - 国内环境自动检测并切换 hf-mirror.com
#   - 如需手动指定镜像: export HF_ENDPOINT=https://hf-mirror.com
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SONGFORMER_DIR="${SCRIPT_DIR}/SongFormer"
CKPTS_DIR="${SONGFORMER_DIR}/ckpts"

# ─── 颜色输出 ────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── 依赖检查 ────────────────────────────────────────────────────────────────

check_prerequisites() {
    local missing=()
    command -v python3 >/dev/null 2>&1 || missing+=("python3")
    command -v pip     >/dev/null 2>&1 || command -v pip3 >/dev/null 2>&1 || missing+=("pip")
    command -v ffmpeg  >/dev/null 2>&1 || missing+=("ffmpeg (optional, needed for audio processing)")
    command -v ffprobe >/dev/null 2>&1 || missing+=("ffprobe (optional, needed for duration detection)")

    if [ ${#missing[@]} -gt 0 ]; then
        warn "以下工具未找到:"
        for m in "${missing[@]}"; do
            echo "  - $m"
        done
        echo ""
    fi
}

# ─── 安装 Python 依赖 ────────────────────────────────────────────────────────

install_deps() {
    info "安装主 Pipeline Python 依赖..."
    pip install -r "${SCRIPT_DIR}/requirements.txt"

    info "安装 SongFormer Python 依赖..."
    pip install -r "${SONGFORMER_DIR}/requirements.txt"

    info "Python 依赖安装完成"
}

# ─── 下载 SongFormer 权重 ────────────────────────────────────────────────────

detect_mirror() {
    if [ -n "${HF_ENDPOINT:-}" ]; then
        echo "$HF_ENDPOINT"
        return
    fi
    if curl -s --max-time 5 "https://huggingface.co" >/dev/null 2>&1; then
        echo "https://huggingface.co"
    else
        warn "huggingface.co 不可达，切换到 hf-mirror.com"
        echo "https://hf-mirror.com"
    fi
}

download_file() {
    local url="$1" dest="$2" expected_md5="${3:-}"
    if [ -f "$dest" ]; then
        if [ -n "$expected_md5" ]; then
            local actual_md5
            actual_md5=$(md5sum "$dest" 2>/dev/null | awk '{print $1}') || true
            if [ "$actual_md5" = "$expected_md5" ]; then
                info "已存在且校验通过，跳过: $(basename "$dest")"
                return 0
            else
                warn "文件已存在但 MD5 不匹配，重新下载: $(basename "$dest")"
            fi
        else
            info "已存在，跳过: $(basename "$dest")"
            return 0
        fi
    fi

    mkdir -p "$(dirname "$dest")"
    info "下载: $(basename "$dest")"
    echo "  URL: $url"

    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$dest" "$url"
    else
        curl -L --progress-bar -o "$dest" "$url"
    fi

    if [ -n "$expected_md5" ]; then
        local actual_md5
        actual_md5=$(md5sum "$dest" 2>/dev/null | awk '{print $1}') || true
        if [ "$actual_md5" != "$expected_md5" ]; then
            error "MD5 校验失败: $(basename "$dest")"
            error "  期望: $expected_md5"
            error "  实际: $actual_md5"
            return 1
        fi
        info "MD5 校验通过: $(basename "$dest")"
    fi
}

download_weights() {
    local base_url
    base_url=$(detect_mirror)
    info "使用下载源: $base_url"
    echo ""

    info "下载 SongFormer 权重到 ${CKPTS_DIR}/ ..."
    echo ""

    download_file \
        "${base_url}/minzwon/MusicFM/resolve/main/msd_stats.json" \
        "${CKPTS_DIR}/MusicFM/msd_stats.json" \
        "75ab2e47b093e07378f7f703bdb82c14"

    download_file \
        "${base_url}/minzwon/MusicFM/resolve/main/pretrained_msd.pt" \
        "${CKPTS_DIR}/MusicFM/pretrained_msd.pt" \
        "df930aceac8209818556c4a656a0714c"

    download_file \
        "${base_url}/ASLP-lab/SongFormer/resolve/main/SongFormer.safetensors" \
        "${CKPTS_DIR}/SongFormer.safetensors" \
        "5a24800e12ab357744f8b47e523ba3e6"

    echo ""
    info "SongFormer 权重下载完成"
}

# ─── 主程序 ───────────────────────────────────────────────────────────────────

usage() {
    cat <<'EOF'
Music Data Process Pipeline — 一键环境安装

用法:
  bash setup.sh              安装全部依赖 + 下载 SongFormer 权重
  bash setup.sh --weights    仅下载 SongFormer 权重 (~8 GB)
  bash setup.sh --deps       仅安装 Python 依赖
  bash setup.sh --check      检查环境依赖
  bash setup.sh --help       显示帮助
EOF
}

main() {
    echo "========================================="
    echo "  Music Data Process Pipeline Setup"
    echo "========================================="
    echo ""

    case "${1:-all}" in
        --weights|-w)
            check_prerequisites
            download_weights
            ;;
        --deps|-d)
            check_prerequisites
            install_deps
            ;;
        --check|-c)
            check_prerequisites
            info "Python: $(python3 --version 2>&1 || echo 'not found')"
            info "pip: $(pip --version 2>&1 || pip3 --version 2>&1 || echo 'not found')"
            info "ffmpeg: $(ffmpeg -version 2>&1 | head -1 || echo 'not found')"
            echo ""
            if [ -f "${CKPTS_DIR}/SongFormer.safetensors" ]; then
                info "SongFormer 权重: 已下载"
            else
                warn "SongFormer 权重: 未下载 (运行 bash setup.sh --weights)"
            fi
            ;;
        all|--all|-a)
            check_prerequisites
            install_deps
            echo ""
            download_weights
            ;;
        --help|-h)
            usage
            ;;
        *)
            error "未知参数: $1"
            usage
            exit 1
            ;;
    esac

    echo ""
    echo "========================================="
    info "完成!"
    echo "========================================="
}

main "$@"
