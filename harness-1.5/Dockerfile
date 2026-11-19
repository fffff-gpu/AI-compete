FROM python:3.10-slim

# 1.5：只装官方 Harness SDK 的入口协议（contest.harness.workspace.v2）。
# 镜像里没有数据、没有权重、没有密钥：训练从平台给的父检查点续，预测从 /model 读。
WORKDIR /app

ARG ZHIXUN_VERSION=1.5
ARG ZHIXUN_BUILD_SHA=unknown
LABEL org.opencontainers.image.title="zhixun-harness" \
      org.opencontainers.image.version="$ZHIXUN_VERSION" \
      org.opencontainers.image.revision="$ZHIXUN_BUILD_SHA" \
      org.opencontainers.image.description="工业模型自进化 Harness：workspace v2 SDK 入口 + 数据挖掘配方"

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH=/app \
    CONTEST_SUBMISSION=contest_submission

# 依赖只有 numpy：优先离线 wheel，容器内访问 PyPI 在本机实测会挂死。
# 用 --mount=type=bind 而不是 COPY，wheel 不写进镜像层。
ARG ZHIXUN_PIP_MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
COPY requirements.txt /app/requirements.txt
RUN --mount=type=bind,source=_wheels,target=/wheels \
    if ls /wheels/*.whl >/dev/null 2>&1; then \
      echo "[build] numpy offline install (/wheels)"; \
      pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt; \
    else \
      echo "[build] no wheel, online install ($ZHIXUN_PIP_MIRROR)"; \
      pip install --no-cache-dir -i "$ZHIXUN_PIP_MIRROR" \
          --trusted-host pypi.tuna.tsinghua.edu.cn -r /app/requirements.txt; \
    fi \
 && python -c "import numpy; print('[build] numpy', numpy.__version__)"

# 官方入口模板（未改动）→ /app/harness.py；SDK 包与我们的模块同层可 import。
COPY sdk/harness.py /app/harness.py
COPY sdk/contest_harness /app/contest_harness
COPY sdk/rulkit /app/rulkit
COPY sdk/contest_submission.py /app/contest_submission.py

# 构建期自证入口声明能被平台读到（describe 会 import contest_submission）。
RUN python3 /app/harness.py describe \
 && python3 /app/harness.py --version \
 && python3 -c "import contest_submission, rulkit.model; print('[build] submission import ok')"

# 平台挂载：/data（公开投影，只读）、/previous（父产物，只读）、/state（原生状态，只读）、
# /model（预测时的产物，只读）、/workspace（本轮可写）。镜像自身不落任何数据。
VOLUME ["/workspace"]

ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]