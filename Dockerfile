# TensorTrade强化学习交易系统 Docker镜像
# 基于Ubuntu 20.04，包含Python 3.8+ 和所有必要依赖

FROM ubuntu:20.04

# 设置非交互模式和时区
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    software-properties-common \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# 设置Python别名
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 升级pip和安装基础包
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖（分层安装以优化缓存）
# 首先安装基础数据科学库
RUN pip install --no-cache-dir \
    numpy==1.21.6 \
    pandas==1.3.5 \
    scipy==1.7.3 \
    scikit-learn==1.0.2

# 安装深度学习框架
RUN pip install --no-cache-dir \
    tensorflow==2.7.0 \
    torch==1.10.0 \
    torchvision==0.11.1

# 安装Ray和强化学习库
RUN pip install --no-cache-dir \
    ray[rllib]==1.8.0 \
    gym==0.21.0

# 安装TensorTrade
RUN pip install --no-cache-dir \
    tensortrade==1.0.3

# 安装其他依赖
RUN pip install --no-cache-dir \
    yfinance==0.1.87 \
    ta==0.10.2 \
    plotly==5.11.0 \
    dash==2.7.0 \
    joblib==1.2.0 \
    requests==2.28.1 \
    websockets==10.4 \
    aiohttp==3.8.3 \
    fastapi==0.88.0 \
    uvicorn==0.20.0 \
    pytest==7.2.0

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p logs data_cache models results reports configs

# 设置Python路径
ENV PYTHONPATH=/app:$PYTHONPATH

# 设置环境变量
ENV RAY_DISABLE_IMPORT_WARNING=1
ENV RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# 创建非root用户
RUN useradd -m -u 1000 trader && \
    chown -R trader:trader /app

# 切换到非root用户
USER trader

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import sys; import ray; import tensortrade; sys.exit(0)" || exit 1

# 暴露端口
EXPOSE 8000 8265 6379

# 默认命令
CMD ["python", "main.py", "--help"]