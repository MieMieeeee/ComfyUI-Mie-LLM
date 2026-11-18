"""发布后催一下国内 PyPI 镜像: 让新版本尽快被镜像收录.

**为什么需要这一步**
国内 PyPI 镜像 (阿里云 / 华为云 / 清华) 对 PyPI 是**镜像同步**, 不是实时转发.
新发布的版本通常要等它们自己的同步周期才出现 —— 期间用户按
``requirements.txt`` 里的版本号去装, 会拿到 "No matching distribution found".

实测 (2026-11-18): 这些镜像**有按需回源机制** —— 请求一个尚未收录的包的
simple index, 会触发它们去 PyPI 拉取, 约 1 分钟内收录 (受控实验: 15:34:47
请求三个包的 index, 15:36:12 再看, 华为云上三个包全变成了 0.2.0, 而请求前
只有 0.1.x).

所以**发布流程里加一步「请求一下镜像」** 就能把等待从「同步周期」压到「约一分
钟」. 这件事属于**发布侧**: 装包的人 (启动器 / 引擎) 不该去请求镜像 —— 那是出
网方的事, 而且一次请求只能催自己装的那些包.

用法
----
    python tools/notify_mirrors.py                     # 用本仓库 pyproject 的包名
    python tools/notify_mirrors.py comfyui-mie-craft   # 指定包名
    python tools/notify_mirrors.py --wait 180          # 轮询到收录为止 (秒)
    python tools/notify_mirrors.py --check 0.2.0       # 只检查某个版本在不在

发布清单里加一行::

    python tools/notify_mirrors.py --wait 300

只依赖标准库 (不引 requests), 因为它要在任何一台装了 python 的机器上能跑.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

#: 国内 PyPI 镜像的 simple index 根地址.
MIRRORS = {
    "aliyun": "https://mirrors.aliyun.com/pypi/simple",
    "huaweicloud": "https://repo.huaweicloud.com/repository/pypi/simple",
    "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
}

#: PyPI 本体 (对照用).
PYPI = "https://pypi.org/simple"

_UA = "pip/24.0"


def package_name(default: str = "") -> str:
    """从本仓库的 pyproject.toml 读包名 (读不到就用 default)."""
    pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return default
    match = re.search(r'^name\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else default


def _fetch(url: str, timeout: float = 20.0) -> str | None:
    """取一个 URL 的正文; 失败返回 None."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": _UA,
                          "Accept": "text/html, application/vnd.pypi.simple.v1+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def versions_on(index_url: str, pkg: str) -> set:
    """某个镜像上该包已收录的版本集合."""
    body = _fetch("%s/%s/" % (index_url.rstrip("/"), pkg))
    if not body:
        return set()
    found = set()
    # PEP 691 JSON (PyPI 新版默认)
    stripped = body.lstrip()
    if stripped.startswith("{"):
        try:
            for item in json.loads(body).get("files", []):
                name = str(item.get("filename", ""))
                match = re.match(r"^[^-]+-(\d[^-]*?)(\.tar\.gz|-py|\.whl|-\d)", name)
                if match:
                    found.add(match.group(1))
        except Exception:
            pass
        return found
    # 老式 HTML: <a href="...">comfyui_mie_craft-0.2.0-py3-none-any.whl</a>
    for match in re.finditer(r"-\d[\w.\-+!]*?(?:-py\d[^-]*?-none-any\.whl"
                             r"|-none-any\.whl|\.tar\.gz|\.zip)", body):
        token = match.group(0)[1:]
        token = re.sub(r"(-py\d.*|-none-any\.whl|\.tar\.gz|\.zip)$", "", token)
        found.add(token)
    return found


def notify(pkg: str, mirrors: dict, check_version: str = "") -> dict:
    """对每个镜像请求一次该包的 index (触发回源), 并报告目标版本在不在.

    返 {镜像名: {"ok": bool, "has_version": bool, "versions": [...]}}。
    """
    result = {}
    for name, index_url in mirrors.items():
        versions = versions_on(index_url, pkg)
        # 再请求一次: 刚才那次就是触发回源的那次 (异步, 收录要等一会儿)
        _fetch("%s/%s/" % (index_url.rstrip("/"), pkg))
        ok = bool(versions)
        has = bool(check_version) and check_version in versions
        result[name] = {"ok": ok, "has_version": has,
                         "versions": sorted(versions)[-3:] if versions else []}
    return result


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="请求国内 PyPI 镜像, 触发它们收录刚发布的版本")
    parser.add_argument("package", nargs="?", default="",
                        help="包名 (默认读本仓库 pyproject.toml)")
    parser.add_argument("--wait", type=int, default=0, metavar="SEC",
                        help="轮询到目标版本被收录为止 (默认不等, 只报一次)")
    parser.add_argument("--check", default="", metavar="VER",
                        help="要检查的版本号 (默认只报告收录情况)")
    parser.add_argument("--mirror", action="append", default=[],
                        help="只对指定镜像操作 (可重复; 默认全部)")
    args = parser.parse_args(argv)

    pkg = (args.package or package_name() or "").strip()
    if not pkg:
        print("不知道包名: 请传参数或确认 pyproject.toml 里有 name", file=sys.stderr)
        return 2

    mirrors = dict(MIRRORS)
    if args.mirror:
        wanted = {m.strip().lower() for m in args.mirror}
        mirrors = {k: v for k, v in MIRRORS.items() if k in wanted}
        if not mirrors:
            print("未知镜像: %s (可选: %s)" % (", ".join(wanted),
                                           ", ".join(MIRRORS)), file=sys.stderr)
            return 2

    version = args.check
    if not version:
        # 没指定就取 PyPI 上的最新版作为"应该被收录"的版本
        pypi_versions = versions_on(PYPI, pkg)
        version = sorted(pypi_versions)[-1] if pypi_versions else ""
        if not version:
            print("PyPI 上找不到 %s —— 先确认发布成功了吗?" % pkg, file=sys.stderr)
            return 1

    print("包 %s · 目标版本 %s" % (pkg, version))
    deadline = time.time() + max(0, args.wait)
    while True:
        result = notify(pkg, mirrors, version)
        done = True
        for name, info in result.items():
            mark = "✓" if info["has_version"] else ("·" if info["ok"] else "✗")
            print("  %-12s %s  %s" % (name, mark,
                                      ("已收录 " + ", ".join(info["versions"]))
                                      if info["versions"] else "取不到版本列表"))
            if not info["has_version"]:
                done = False
        if done or time.time() >= deadline:
            break
        print("  … 还没收录, 30 秒后再看 (回源大约 1 分钟)")
        time.sleep(30)

    print("\n提示: 镜像收录是异步的, 刚触发的那次请求不会立刻带上新版本。")
    print("      若某个镜像仍未收录, 过一两分钟再跑一次本脚本即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
