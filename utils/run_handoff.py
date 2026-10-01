"""
运行交接摘要（Run Handoff）
===========================

设计目标（2026-09-15）：
> 每次运行结束，把需要看的信息**同时写到文件 + 显示在控制台**，便于记录与复核。

**解决什么痛点**：一次训练刷几百行日志，其中真正要看的只有几行；
复制错了、漏了、或者张冠李戴（拿 old run 的日志当成新 run 的），
都会让分析跑偏。本模块让每个脚本在收尾时输出一段**固定格式、自包含**的摘要块：

    ######## HANDOFF-BEGIN train_tongue_region ########
    tag      : train_tongue_region
    time     : 2026-09-15 17:20:33
    argv     : training/train_tongue_region.py --tongue_root ... --seed 42
    versions : PROTO=2026-09-15 MATRIX=2026-09-15 B2=2026-09-14c
    script   : training/train_tongue_region.py  21406B  mtime 2026-09-15 17:02:11  sha1=9c1d4e77
    ---------------- 核心指标 ----------------
    macro_f1(valid) : 0.3421
    ...
    artifacts:
      runs/tongue_region/best_model.pt
    ######## HANDOFF-END train_tongue_region ########

控制台上这段被 `#` 边框包住，**从 BEGIN 复制到 END** 复制完整即可，
文件同时落在 `<out_dir>/handoff_<tag>.txt`。

设计要点
--------
1. **崩溃也要出摘要**：用 `H.run(main)` 代替 `main()`，异常会被捕获、
   记录 traceback 尾部，然后照样打印 + 落盘 —— 跑崩的那一刻恰恰最需要信息。
2. **自动采集**：时间、argv、cwd、主机名、各模块版本号（PROTO/MATRIX/B2/PROBE），
   以及**主脚本指纹**（`script` 行的 sha1 前 8 位），不依赖使用者记得填。
3. **能验明"脚本到底传上去没有"**（2026-09-15 补）：`versions` 行只反映模型/矩阵口径，
   **反映不出脚本自身的新旧**。本地改好、服务器留旧版时摘要照样"看起来正常"。
   比对方法：本地与服务器各跑 `sha1sum <脚本>`，与摘要 `script` 行一致才说明是最新版。
   若摘要里**根本没有 `script` 行** → 连 `utils/run_handoff.py` 都是旧版，需一并重传。
4. **零依赖**：只用标准库，没装 torch 也能 import（compare_runs 这类纯分析脚本要用）。

用法
----
    from utils.run_handoff import Handoff

    H = Handoff("extract_prototypes_v2")          # 模块级单例

    def main():
        ...
        H.kv("best_strategy", best_strat)          # 关键点随手记
        H.section("各病性支撑图数")
        H.kv_dict(dict(zip(SYNDROMES_ZH, meta["n_images"])))
        H.artifact(rp)
        return 0

    if __name__ == "__main__":
        raise SystemExit(H.run(main))              # 自动 flush（含异常路径）

指定落盘目录：`Handoff("xxx", out_dir=runs_dir)`，或中途 `H.set_out_dir(...)`。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import io
import os
import platform
import socket
import sys
import traceback
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

__all__ = ["Handoff", "make"]

# 自动探测的模块版本常量（(模块, 属性, 简称)）
_VERSION_PROBES = [
    ("models.tongue_vision_branch", "PROTO_VERSION", "PROTO"),
    ("models.tongue_label_mapping", "MATRIX_VERSION", "MATRIX"),
    ("models.syndrome_proto_attention", "B2_VERSION", "B2"),
    ("eval.encoder_probe", "PROBE_VERSION", "PROBE"),
]

_BAR = "#" * 8
_DEFAULT_DIR = "_handoff"


def _script_fingerprint() -> dict:
    """主脚本指纹：路径 / 字节数 / mtime / sha1 前 8 位。

    **为什么需要它**（2026-09-15 教训，连栽两次）：摘要里的 `versions :` 行反映的是
    **模型/矩阵口径**（PROTO/MATRIX/B2/PROBE），却**反映不出这个脚本本身是新版还是旧版**。
    本地改好脚本、服务器上仍是旧文件时，摘要照样打印 `PROTO=2026-09-15`，
    看起来"一切正常"，实际跑的是旧逻辑 —— 只能靠人工比对输出里多了哪几行，很不可靠。
    有了 sha1，`sha1sum 脚本` 与摘要里的值一比即知是否传成功。
    """
    p = (sys.argv[0] if sys.argv else "") or ""
    try:
        if not p or not os.path.isfile(p):
            return {}
        fp = Path(p).resolve()
        st = fp.stat()
        try:
            rel = os.path.relpath(fp)
            if rel.startswith(".."):
                rel = str(fp)
        except Exception:
            rel = str(fp)
        return {
            "path": rel,
            "abs": str(fp),
            "size": int(st.st_size),
            "mtime": _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "sha1": hashlib.sha1(fp.read_bytes()).hexdigest()[:8],
        }
    except Exception:
        return {}


def _dwidth(s: Any) -> int:
    """终端显示宽度：中文/全角算 2，其余算 1。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in str(s))


def _pad(s: Any, width: int) -> str:
    s = str(s)
    return s + " " * max(0, width - _dwidth(s))


class Handoff:
    """收集一次运行的关键信息，收尾时打印 + 落盘。"""

    def __init__(self, tag: str, out_dir: Optional[str | Path] = None,
                 echo: bool = True, auto_versions: bool = True):
        self.tag = tag
        self.out_dir = Path(out_dir) if out_dir else None
        self.echo = echo
        self._sections: list[tuple[str, list[str]]] = []   # [(标题, 行列表)]
        self._cur_title = "概要"
        self._cur: list[str] = []
        self._artifacts: list[str] = []
        self._warnings: list[str] = []
        self._failures: list[str] = []
        self._flushed_paths: list[Path] = []
        self._start = _dt.datetime.now()
        self.enabled = True   # 设 False 可整体关闭（如 --selftest 路径）
        self.tee = False      # 设 True 则把整个控制台输出也附进摘要（分析类脚本用）
        self._buf: Optional[io.StringIO] = None
        self.versions = self._probe_versions() if auto_versions else {}
        # 主脚本指纹（防"服务器上还是旧版脚本"）：摘要里多打一行 script
        self.script_fp = _script_fingerprint() if auto_versions else {}

    # ---------------- 控制台回显捕获 ----------------

    @contextmanager
    def tee_stdout(self):
        """同时写真实 stdout 和内存缓冲。退出后缓冲内容可被 render 附加。"""
        buf = io.StringIO()
        self._buf = buf
        old = sys.stdout

        class _Tee:
            def write(self, s):
                old.write(s)
                buf.write(s)
                return len(s)

            def flush(self):
                try:
                    old.flush()
                except Exception:
                    pass

            def isatty(self):
                return False

        sys.stdout = _Tee()  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = old
            self._buf = buf   # 保留本次捕获，供 render 附加

    # ---------------- 采集 ----------------

    def set_out_dir(self, out_dir: str | Path) -> None:
        """中途设定落盘目录（例如脚本解析完 --output_dir 之后）。"""
        self.out_dir = Path(out_dir)

    def section(self, title: str) -> None:
        self._flush_section()
        self._cur_title = title

    def line(self, text: str = "") -> None:
        self._cur.append(str(text))

    def kv(self, key: str, value: Any) -> None:
        self._cur.append(f"{key} : {value}")

    def kv_dict(self, d: dict, sep: str = "  ") -> None:
        """把 {名: 值} 压成一行（适合 10 个病性这种短表）。"""
        self._cur.append(sep.join(f"{k}={v}" for k, v in d.items()))

    def table(self, headers: Iterable[str], rows: Iterable[Iterable[Any]],
              widths: Optional[Iterable[int]] = None) -> None:
        """等宽表格（按终端显示宽度对齐，中文算 2 列）。"""
        hs = [str(h) for h in headers]
        ws = list(widths) if widths else [max(_dwidth(h), 10) for h in hs]
        self._cur.append("  ".join(_pad(h, w) for h, w in zip(hs, ws)))
        self._cur.append("  ".join("-" * w for w in ws))
        for r in rows:
            self._cur.append("  ".join(_pad(c, w) for c, w in zip(r, ws)))

    def artifact(self, path: str | Path) -> None:
        p = str(path)
        if p not in self._artifacts:
            self._artifacts.append(p)

    def ok(self, msg: str) -> None:
        self._cur.append(f"[OK] {msg}")

    def warn(self, msg: str) -> None:
        m = str(msg)
        if m not in self._warnings:
            self._warnings.append(m)
        self._cur.append(f"[WARN] {m}")

    def fail(self, msg: str) -> None:
        m = str(msg)
        if m not in self._failures:
            self._failures.append(m)
        self._cur.append(f"[FAIL] {m}")

    def _flush_section(self) -> None:
        if self._cur:
            self._sections.append((self._cur_title, self._cur))
            self._cur = []

    # ---------------- 渲染 ----------------

    @staticmethod
    def _probe_versions() -> dict:
        out = {}
        for mod, attr, short in _VERSION_PROBES:
            try:
                m = importlib.import_module(mod)
                out[short] = str(getattr(m, attr, "?"))
            except Exception:
                out[short] = "-"
        return out

    def render(self, exit_code: int = 0, exc_text: str = "") -> str:
        self._flush_section()
        L: list[str] = []
        L.append(f"{_BAR} HANDOFF-BEGIN {self.tag} {_BAR}")
        L.append(f"tag      : {self.tag}")
        L.append(f"time     : {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                 f"  (耗时 {( _dt.datetime.now() - self._start).total_seconds():.1f}s)")
        L.append(f"exit_code: {exit_code}")
        L.append(f"argv     : {' '.join(sys.argv)}")
        L.append(f"cwd      : {os.getcwd()}")
        L.append(f"host     : {socket.gethostname()} / {platform.system()} "
                 f"{platform.release()} / py{sys.version.split()[0]}")
        if self.versions:
            L.append("versions : " + "  ".join(f"{k}={v}" for k, v in self.versions.items()))
        if self.script_fp:
            sf = self.script_fp
            L.append(f"script   : {sf['path']}  {sf['size']}B  "
                     f"mtime {sf['mtime']}  sha1={sf['sha1']}")
        for title, lines in self._sections:
            L.append(f"---------------- {title} ----------------")
            L.extend(lines)
        if self._artifacts:
            L.append("---------------- artifacts ----------------")
            L.extend(f"  {a}" for a in self._artifacts)
        if self._warnings:
            L.append("---------------- warnings ----------------")
            L.extend(f"  {w}" for w in self._warnings)
        if self._failures:
            L.append("---------------- failures ----------------")
            L.extend(f"  {f}" for f in self._failures)
        if exc_text:
            L.append("---------------- traceback(尾部) ----------------")
            L.extend(exc_text.strip().splitlines()[-25:])
        # ---- 可选：附上被捕获的完整控制台输出（H.tee=True 时）----
        if self._buf is not None:
            cap = self._buf.getvalue()
            if cap.strip():
                lines = cap.rstrip().splitlines()
                L.append(f"---------------- 完整控制台输出（共 {len(lines)} 行）----------------")
                if len(lines) > 220:
                    L.append(f"...(前 {len(lines)-200} 行略)...")
                    lines = lines[-200:]
                L.extend(lines)
        L.append("---------------- 复制提示 ----------------")
        L.append(f"请把 BEGIN 到 END 之间这段完整复制（或 cat 下面文件）：")
        if self._flushed_paths:
            L.extend(f"  {p}" for p in self._flushed_paths)
        L.append(f"{_BAR} HANDOFF-END {self.tag} {_BAR}")
        return "\n".join(L)

    def flush(self, exit_code: int = 0, exc_text: str = "") -> list[Path]:
        """打印到控制台 + 落盘。返回写出的文件路径列表。

        落盘策略（2026-09-15）：**双写**
          · 统一目录 `<cwd>/_handoff/` —— 固定名 + 时间戳 + LATEST.txt
            （只需记一个路径：`cat _handoff/LATEST.txt`）
          · 脚本指定的 out_dir（如 runs/xxx/）—— 在该 run 目录留档
        """
        if not self.enabled:
            return []

        stamp = self._start.strftime("%Y%m%d_%H%M%S")
        base = Path(_DEFAULT_DIR)
        dirs: list[Path] = [base]
        if self.out_dir is not None:
            try:
                if Path(self.out_dir).resolve() != base.resolve():
                    dirs.append(Path(self.out_dir))
            except Exception:
                dirs.append(Path(self.out_dir))

        # 写文件的目标（含"最新"指针）
        targets: list[Path] = []
        for d in dirs:
            targets.append(d / f"handoff_{self.tag}.txt")
            targets.append(d / f"handoff_{self.tag}_{stamp}.txt")
        targets.append(base / "LATEST.txt")

        # 控制台"复制提示"里只列最常用的两个，避免刷屏
        self._flushed_paths = [base / f"handoff_{self.tag}.txt", base / "LATEST.txt"]
        text = self.render(exit_code=exit_code, exc_text=exc_text)

        written: list[Path] = []
        for p in targets:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
                written.append(p)
            except Exception as e:
                print(f"[WARN] 摘要写入失败 {p}：{type(e).__name__}: {e}", file=sys.stderr)

        if self.echo:
            print("\n" + text, flush=True)
        return written

    # ---------------- 执行包装 ----------------

    def run(self, fn, *a, **kw) -> int:
        """执行 fn(*a, **kw)，无论成功或异常都输出摘要；返回进程退出码。"""
        code = 0
        exc_text = ""
        try:
            if self.tee:
                with self.tee_stdout():
                    r = fn(*a, **kw)
            else:
                r = fn(*a, **kw)
            if isinstance(r, int):
                code = r
        except SystemExit as e:
            code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
            if code:
                exc_text = "".join(traceback.format_exception_only(type(e), e))
                self.fail(f"SystemExit({code})")
        except BaseException as e:
            code = 1
            exc_text = traceback.format_exc()
            self.fail(f"{type(e).__name__}: {e}")
        try:
            self.flush(exit_code=code, exc_text=exc_text)
        except Exception:
            traceback.print_exc()
        return code


def make(tag: str, out_dir: Optional[str | Path] = None, **kw) -> Handoff:
    return Handoff(tag, out_dir=out_dir, **kw)


# ============================================================
# 自测：python -m utils.run_handoff
# ============================================================

def _selftest() -> int:
    import tempfile

    print("=" * 70)
    print("run_handoff 自测")
    print("=" * 70)
    ok = True

    with tempfile.TemporaryDirectory() as td:
        # 把默认落盘目录也重定向到临时目录，避免测试污染工作区
        global _DEFAULT_DIR
        _saved_default = _DEFAULT_DIR
        _DEFAULT_DIR = str(Path(td) / "_handoff_default")

        try:
            # ---- 1. 正常路径 ----
            H = Handoff("selftest_ok", out_dir=td, echo=False)
            H.kv("matrix_version", "2026-09-15")
            H.section("核心指标")
            H.kv("macro_f1", 0.3421)
            H.kv_dict({"气虚": 4495, "血虚": 189})
            H.section("逐类")
            H.table(["病性", "F1"], [["气虚", 0.81], ["血虚", 0.61]], widths=[8, 8])
            H.artifact("runs/x/best.pt")
            H.warn("示例告警")
            code = H.run(lambda: 0)
            assert code == 0, f"正常路径退出码应为 0，得到 {code}"
            txt = (Path(td) / "handoff_selftest_ok.txt").read_text(encoding="utf-8")
            for token in ["HANDOFF-BEGIN selftest_ok", "HANDOFF-END selftest_ok",
                          "matrix_version : 2026-09-15", "macro_f1 : 0.3421",
                          "气虚=4495", "best.pt", "示例告警", "exit_code: 0",
                          "script   :", "sha1="]:
                if token not in txt:
                    print(f"  ✗ 缺失内容：{token}")
                    ok = False
            # 双写：out_dir 与默认目录都要有
            assert (Path(td) / "handoff_selftest_ok.txt").exists(), "out_dir 落点未生成"
            assert (Path(_DEFAULT_DIR) / "handoff_selftest_ok.txt").exists(), \
                "默认目录落点未生成"
            assert (Path(_DEFAULT_DIR) / "LATEST.txt").exists(), "LATEST.txt 未生成"
            assert list(Path(td).glob("handoff_selftest_ok_*.txt")), "时间戳文件未生成"
            print("  ✓ 正常路径：内容完整、双写落点 + LATEST + 时间戳均已生成")

            # ---- 2. 异常路径（最关键：跑崩了也要有摘要）----
            H2 = Handoff("selftest_fail", out_dir=td, echo=False)
            H2.kv("seed", 42)

            def boom():
                raise ValueError("故意炸一个")

            code2 = H2.run(boom)
            assert code2 == 1, f"异常路径退出码应为 1，得到 {code2}"
            txt2 = (Path(td) / "handoff_selftest_fail.txt").read_text(encoding="utf-8")
            for token in ["exit_code: 1", "ValueError: 故意炸一个", "Traceback", "seed : 42"]:
                if token not in txt2:
                    print(f"  ✗ 异常摘要缺失：{token}")
                    ok = False
            print("  ✓ 异常路径：exit_code=1、traceback 尾部、已知字段均保留")

            # ---- 3. SystemExit(2) 也要被记录 ----
            H3 = Handoff("selftest_sysexit", out_dir=td, echo=False)

            def sysexit():
                raise SystemExit(2)

            assert H3.run(sysexit) == 2, "SystemExit(2) 应透传为退出码 2"
            txt3 = (Path(td) / "handoff_selftest_sysexit.txt").read_text(encoding="utf-8")
            assert "exit_code: 2" in txt3, "SystemExit 未被记录"
            print("  ✓ SystemExit(2)：退出码透传且被记录")

            # ---- 4. tee：完整控制台输出被附进摘要 ----
            H4 = Handoff("selftest_tee", out_dir=td, echo=False)
            H4.tee = True

            def noisy():
                print("这行应该出现在摘要的完整控制台输出里")
                return 0

            H4.run(noisy)
            txt4 = (Path(td) / "handoff_selftest_tee.txt").read_text(encoding="utf-8")
            assert "完整控制台输出" in txt4, "tee 未附加控制台输出"
            assert "这行应该出现在摘要的完整控制台输出里" in txt4, "tee 未捕获 print"
            print("  ✓ tee：控制台输出被完整附进摘要")

            # ---- 5. 版本自动探测（本机缺 torch 时也要能跑）----
            H5 = Handoff("selftest_ver", out_dir=td, echo=False)
            print(f"  ✓ 版本自动探测：{H5.versions}")

            # ---- 6. enabled=False 时完全静默 ----
            H6 = Handoff("selftest_off", out_dir=td, echo=False)
            H6.enabled = False
            H6.run(lambda: 0)
            assert not (Path(td) / "handoff_selftest_off.txt").exists(), \
                "enabled=False 时不应落盘"
            print("  ✓ enabled=False：不落盘、不打印")

            # ---- 7. 落盘目录不可写时不崩 ----
            H7 = Handoff("selftest_bad", out_dir="/nonexistent/path/x", echo=False)
            H7.kv("k", "v")
            H7.flush()
            print("  ✓ 落盘失败时未抛异常（仅告警）")

            # ---- 8. 中文宽度对齐 ----
            from utils.run_handoff import _dwidth
            assert _dwidth("病性") == 4 and _dwidth("abc") == 3
            H8 = Handoff("selftest_width", out_dir=td, echo=False)
            H8.table(["病性", "F1"], [["气虚", 0.81]], widths=[8, 6])
            print(f"  ✓ 中文宽度对齐：_dwidth('病性')={_dwidth('病性')}")

            # ---- 9. 主脚本指纹：argv[0] 不是文件时静默跳过（不崩、不打印空行）----
            _argv0 = sys.argv[0]
            try:
                sys.argv[0] = "-c"
                assert _script_fingerprint() == {}, "argv[0] 非文件时应返回空 dict"
                H9 = Handoff("selftest_fp", out_dir=td, echo=False)
                assert H9.script_fp == {}
                H9.kv("k", "v")
                assert "script   :" not in H9.render(), "无指纹时不应打印 script 行"
                sys.argv[0] = _argv0
                H9b = Handoff("selftest_fp2", out_dir=td, echo=False)
                assert H9b.script_fp.get("sha1"), "真实脚本应能算出 sha1"
                assert H9b.script_fp.get("mtime"), "真实脚本应带 mtime"
                assert "script   :" in H9b.render()
                print(f"  ✓ 主脚本指纹：非文件 argv 静默跳过；"
                      f"真实脚本 sha1={H9b.script_fp['sha1']} "
                      f"mtime={H9b.script_fp['mtime']}")
            finally:
                sys.argv[0] = _argv0
        finally:
            _DEFAULT_DIR = _saved_default

    print("=" * 70)
    print("全部自测通过 ✓" if ok else "有失败项 ✗")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
