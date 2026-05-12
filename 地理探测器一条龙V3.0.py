from pathlib import Path
import csv
import queue
import threading
import traceback
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import geopandas as gpd
import jenkspy
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import font_manager
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as transform_coords


DEFAULT_ROOT = ""
DEFAULT_FACTOR_DIR = ""
DEFAULT_SHP = ""
DEFAULT_Y_TIF = ""
DEFAULT_CLIP_DIR = ""
DEFAULT_CLASS_DIR = ""
DEFAULT_OUTPUT_DIR = ""

CLIP_NODATA = -9999.0
CLASS_NODATA = 0
DEFAULT_JENKS_SAMPLE_LIMIT = 10_000
DEFAULT_SAMPLE_INTERVAL_M = 1000
RANDOM_SEED = 260
ABNORMAL_ABS_LIMIT = 1e20

HEATMAP_CMAPS = {
    "黄红渐变 YlOrRd": "YlOrRd",
    "橙红渐变 OrRd": "OrRd",
    "火焰色 inferno": "inferno",
    "岩浆色 magma": "magma",
    "洋红黄绿 Spectral": "Spectral_r",
    "蓝绿黄 viridis": "viridis",
    "蓝紫红 plasma": "plasma",
    "红蓝发散 coolwarm": "coolwarm",
}


def get_chinese_font():
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\Deng.ttf",
        r"C:\Windows\Fonts\Dengb.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    ]
    for candidate in candidates:
        font_path = Path(candidate)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            return font_manager.FontProperties(fname=str(font_path))
    for family in ("Microsoft YaHei", "SimHei", "SimSun", "DengXian", "PingFang SC", "Songti SC"):
        try:
            font_path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        if font_path and Path(font_path).exists():
            return font_manager.FontProperties(fname=font_path)
    return None


def factor_resampling(path):
    if "土地利用" in path.stem:
        return Resampling.nearest
    return Resampling.bilinear


def natural_breaks(values, n_classes, sample_limit, log):
    values = values[np.isfinite(values)]
    unique = np.unique(values)
    if unique.size == 0:
        raise ValueError("没有可用于自然断点分类的有效像元。")
    if unique.size <= n_classes:
        breaks = [float(unique[0])]
        breaks.extend(float(v) for v in unique[1:])
        while len(breaks) < n_classes + 1:
            breaks.append(float(unique[-1]))
        return breaks, "unique-values"

    if values.size > sample_limit:
        log(f"有效像元 {values.size:,} 个，抽样 {sample_limit:,} 个计算 Jenks 自然断点。")
        rng = np.random.default_rng(RANDOM_SEED)
        sample = rng.choice(values, size=sample_limit, replace=False)
        sample.sort()
        return jenkspy.jenks_breaks(sample, n_classes), f"jenks-sampled-{sample_limit}"

    sorted_values = np.sort(values)
    return jenkspy.jenks_breaks(sorted_values, n_classes), "jenks-full"


def q_stat(y, category, valid_mask):
    valid = valid_mask & (category > 0) & np.isfinite(y)
    yy = y[valid]
    cc = category[valid]
    n = yy.size
    if n == 0:
        return np.nan, 0, 0

    total_var = np.var(yy)
    if total_var == 0:
        return np.nan, int(n), int(np.unique(cc).size)

    within = 0.0
    for value in np.unique(cc):
        group = yy[cc == value]
        within += group.size * np.var(group)
    q = 1.0 - within / (n * total_var)
    return float(max(0.0, min(1.0, q))), int(n), int(np.unique(cc).size)


def interaction_type(q1, q2, q12):
    eps = 1e-10
    qmin, qmax, qsum = min(q1, q2), max(q1, q2), q1 + q2
    if q12 < qmin - eps:
        return "非线性减弱"
    if q12 < qmax - eps:
        return "单因子减弱"
    if abs(q12 - qsum) <= eps:
        return "相互独立"
    if q12 < qsum - eps:
        return "双因子增强"
    return "非线性增强"


def format_interval(interval_m):
    if float(interval_m).is_integer():
        return str(int(interval_m))
    return str(interval_m).replace(".", "p")


def safe_cmap_name(label_or_name):
    return HEATMAP_CMAPS.get(label_or_name, label_or_name)


class GeoDetectorProcessor:
    def __init__(self, config, log):
        self.factor_dir = Path(config["factor_dir"])
        self.shp_path = Path(config["shp_path"])
        self.y_tif = Path(config["y_tif"])
        self.clip_dir = Path(config["clip_dir"])
        self.class_dir = Path(config["class_dir"])
        self.output_dir = Path(config["output_dir"])
        self.n_classes = int(config["n_classes"])
        self.sample_limit = int(config["sample_limit"])
        self.sample_interval_m = float(config["sample_interval_m"])
        self.heatmap_cmap = safe_cmap_name(config["heatmap_cmap"])
        self.skip_existing = bool(config["skip_existing"])
        self.log = log
        self._grid = None
        self._mask = None

    def ensure_dirs(self):
        for path in (self.clip_dir, self.class_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)

    def factor_paths(self):
        paths = sorted(self.factor_dir.glob("*.tif"))
        if not paths:
            raise ValueError(f"因子目录中没有 tif 文件: {self.factor_dir}")
        return paths

    def clip_paths(self):
        paths = sorted(self.clip_dir.glob("*_clip.tif"))
        if not paths:
            raise ValueError(f"裁剪目录中没有 *_clip.tif 文件: {self.clip_dir}")
        return paths

    def class_paths(self):
        paths = sorted(self.class_dir.glob(f"*_{self.n_classes}class.tif"))
        if not paths and self.n_classes == 5:
            paths = sorted(self.class_dir.glob("*_5class.tif"))
        if not paths:
            raise ValueError(f"分类目录中没有 *_{self.n_classes}class.tif 文件: {self.class_dir}")
        return paths

    def sample_csv_path(self):
        interval = format_interval(self.sample_interval_m)
        return self.output_dir / f"渔网点XY提取值_{interval}m.csv"

    def target_grid(self):
        if self._grid is None:
            with rasterio.open(self.y_tif) as src:
                self._grid = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                    "profile": src.profile.copy(),
                }
        return self._grid

    def shapes_in_target_crs(self):
        grid = self.target_grid()
        gdf = gpd.read_file(self.shp_path)
        if gdf.crs is None:
            raise ValueError(f"边界 shp 没有 CRS: {self.shp_path}")
        if gdf.crs != grid["crs"]:
            self.log(f"边界 CRS {gdf.crs} 与 Y 栅格 CRS {grid['crs']} 不一致，正在重投影。")
            gdf = gdf.to_crs(grid["crs"])
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        if gdf.empty:
            raise ValueError("边界 shp 没有可用几何。")
        gdf["geometry"] = gdf.geometry.buffer(0)
        return list(gdf.geometry)

    def mask(self):
        if self._mask is None:
            grid = self.target_grid()
            shapes = self.shapes_in_target_crs()
            self._mask = rasterize(
                [(geom, 1) for geom in shapes],
                out_shape=(grid["height"], grid["width"]),
                transform=grid["transform"],
                fill=0,
                dtype="uint8",
                all_touched=False,
            ).astype(bool)
        return self._mask

    def clip_one_factor(self, path):
        grid = self.target_grid()
        mask = self.mask()
        out_path = self.clip_dir / f"{path.stem}_clip.tif"
        if out_path.exists() and self.skip_existing:
            self.log(f"跳过已有裁剪文件: {out_path.name}")
            return out_path

        with rasterio.open(path) as src:
            vrt_options = {
                "crs": grid["crs"],
                "transform": grid["transform"],
                "width": grid["width"],
                "height": grid["height"],
                "resampling": factor_resampling(path),
                "nodata": src.nodata,
            }
            with WarpedVRT(src, **vrt_options) as vrt:
                data = vrt.read(1, out_dtype="float32")
                vrt_nodata = vrt.nodata

        invalid = ~mask | ~np.isfinite(data)
        if vrt_nodata is not None and np.isfinite(vrt_nodata):
            invalid |= np.isclose(data, vrt_nodata)
        elif vrt_nodata is not None:
            invalid |= data == vrt_nodata

        data = data.astype("float32", copy=False)
        data[invalid] = CLIP_NODATA

        profile = grid["profile"].copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=CLIP_NODATA,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        return out_path

    def run_clip(self):
        self.ensure_dirs()
        paths = self.factor_paths()
        self.log(f"开始数据裁剪，共 {len(paths)} 个因子 tif。")
        outputs = []
        for idx, path in enumerate(paths, 1):
            self.log(f"[{idx}/{len(paths)}] 裁剪 {path.name}")
            outputs.append(self.clip_one_factor(path))
        self.log(f"数据裁剪完成，输出目录: {self.clip_dir}")
        return outputs

    def classify_one(self, path):
        out_path = self.class_dir / f"{path.stem.replace('_clip', '')}_{self.n_classes}class.tif"
        if out_path.exists() and self.skip_existing:
            self.log(f"跳过已有分类文件: {out_path.name}")
            return out_path, None

        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile.copy()
            nodata = src.nodata

        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata
        values = data[valid].astype("float64")
        if values.size == 0:
            raise ValueError(f"{path.name} 裁剪后没有有效像元。")

        breaks, method = natural_breaks(values, self.n_classes, self.sample_limit, self.log)
        inner = np.array(breaks[1:-1], dtype="float64")
        classes = np.full(data.shape, CLASS_NODATA, dtype="uint8")
        classes[valid] = np.searchsorted(inner, data[valid], side="right").astype("uint8") + 1

        profile.update(
            dtype="uint8",
            nodata=CLASS_NODATA,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(classes, 1)

        row = {
            "factor": path.stem.replace("_clip", ""),
            "valid_cells": int(values.size),
            "jenks_method": method,
        }
        for i, value in enumerate(breaks):
            suffix = "min" if i == 0 else "max" if i == self.n_classes else str(i)
            row[f"break_{suffix}"] = value
        return out_path, row

    def run_classify(self):
        self.ensure_dirs()
        clip_paths = self.clip_paths()
        self.log(f"开始五级自然断点分类，共 {len(clip_paths)} 个裁剪 tif。")
        outputs = []
        break_rows = []
        for idx, path in enumerate(clip_paths, 1):
            self.log(f"[{idx}/{len(clip_paths)}] 分类 {path.name}")
            out_path, row = self.classify_one(path)
            outputs.append(out_path)
            if row is not None:
                break_rows.append(row)
        if break_rows:
            pd.DataFrame(break_rows).to_csv(
                self.output_dir / "五级分类自然断点.csv",
                index=False,
                encoding="utf-8-sig",
            )
            self.log("已输出五级分类自然断点.csv")
        self.log(f"五级分类完成，输出目录: {self.class_dir}")
        return outputs

    def read_y(self):
        with rasterio.open(self.y_tif) as src:
            y = src.read(1).astype("float64")
            nodata = src.nodata
        valid = self.mask() & np.isfinite(y)
        if nodata is not None:
            valid &= y != nodata
        return y, valid

    def load_class_arrays(self):
        arrays = {}
        common_profile = None
        for path in self.class_paths():
            with rasterio.open(path) as src:
                arr = src.read(1)
                profile = src.profile.copy()
            if common_profile is None:
                common_profile = profile
            name = path.stem.replace(f"_{self.n_classes}class", "").replace("_5class", "")
            arrays[name] = arr
        return arrays

    def common_valid_mask(self, y_valid, factor_arrays):
        common_valid = y_valid.copy()
        for arr in factor_arrays.values():
            common_valid &= arr > 0
        return common_valid

    def metric_boundary(self):
        gdf = gpd.read_file(self.shp_path)
        if gdf.crs is None:
            raise ValueError(f"边界 shp 没有 CRS: {self.shp_path}")
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
        if gdf.empty:
            raise ValueError("边界 shp 没有可用几何。")
        gdf["geometry"] = gdf.geometry.buffer(0)

        candidates = ["ESRI:102025"]
        if hasattr(gdf, "estimate_utm_crs"):
            try:
                utm = gdf.estimate_utm_crs()
                if utm is not None:
                    candidates.append(utm)
            except Exception:
                pass
        candidates.append("EPSG:3857")

        last_error = None
        for crs in candidates:
            try:
                projected = gdf.to_crs(crs)
                self.log(f"采样渔网使用米制投影: {projected.crs}")
                return projected, projected.crs
            except Exception as exc:
                last_error = exc
        raise ValueError(f"无法为采样渔网建立米制投影: {last_error}")

    def sampling_grid_points(self):
        if self.sample_interval_m <= 0:
            raise ValueError("采样点间隔必须大于 0 米。")

        projected, metric_crs = self.metric_boundary()
        minx, miny, maxx, maxy = projected.total_bounds
        interval = self.sample_interval_m
        origin_x = np.floor(minx / interval) * interval
        origin_y = np.ceil(maxy / interval) * interval
        width = int(np.ceil((maxx - origin_x) / interval))
        height = int(np.ceil((origin_y - miny) / interval))
        cell_count = width * height
        if width <= 0 or height <= 0:
            raise ValueError("边界范围异常，无法生成采样渔网。")
        if cell_count > 50_000_000:
            raise ValueError(
                f"当前间隔会生成约 {cell_count:,} 个候选格网，请把采样间隔调大后再试。"
            )

        grid_transform = from_origin(origin_x, origin_y, interval, interval)
        shapes = [(geom, 1) for geom in projected.geometry]
        inside = rasterize(
            shapes,
            out_shape=(height, width),
            transform=grid_transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        ).astype(bool)
        grid_rows, grid_cols = np.where(inside)
        if grid_rows.size == 0:
            raise ValueError("采样渔网在边界内没有生成任何点。")
        self.log(f"边界内候选采样点: {grid_rows.size:,} 个，间隔: {interval:g}m x {interval:g}m")
        return grid_rows, grid_cols, grid_transform, metric_crs

    def grid_index_from_coords(self, xs, ys):
        transform = self.target_grid()["transform"]
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            rows, cols = rowcol(transform, xs, ys)
            return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
        cols = np.floor((xs - transform.c) / transform.a).astype(np.int64)
        rows = np.floor((ys - transform.f) / transform.e).astype(np.int64)
        return rows, cols

    def valid_y_mask(self, values, nodata):
        valid = np.isfinite(values) & (np.abs(values) < ABNORMAL_ABS_LIMIT)
        if nodata is not None:
            if np.isfinite(nodata):
                valid &= ~np.isclose(values, nodata)
            else:
                valid &= values != nodata
        return valid

    def write_sample_table(self):
        out_csv = self.sample_csv_path()
        grid = self.target_grid()
        y_data, _ = self.read_y()
        with rasterio.open(self.y_tif) as src:
            y_nodata = src.nodata
        factor_arrays = self.load_class_arrays()
        names = list(factor_arrays.keys())

        grid_rows, grid_cols, sample_transform, metric_crs = self.sampling_grid_points()
        target_crs = grid["crs"]
        header = ["sample_id", "grid_row", "grid_col", "x", "y", "Y"] + names

        total_candidates = int(grid_rows.size)
        kept = 0
        dropped = 0
        chunk = 100_000
        with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for start in range(0, total_candidates, chunk):
                end = min(start + chunk, total_candidates)
                sample_r = grid_rows[start:end]
                sample_c = grid_cols[start:end]
                x_m = sample_transform.c + sample_transform.a * (sample_c + 0.5)
                y_m = sample_transform.f + sample_transform.e * (sample_r + 0.5)
                if str(metric_crs) == str(target_crs):
                    xs, ys = x_m, y_m
                else:
                    xs_list, ys_list = transform_coords(metric_crs, target_crs, x_m.tolist(), y_m.tolist())
                    xs = np.asarray(xs_list, dtype="float64")
                    ys = np.asarray(ys_list, dtype="float64")

                raster_rows, raster_cols = self.grid_index_from_coords(xs, ys)
                valid = (
                    (raster_rows >= 0)
                    & (raster_rows < grid["height"])
                    & (raster_cols >= 0)
                    & (raster_cols < grid["width"])
                )

                y_values = np.full(sample_r.shape, np.nan, dtype="float64")
                y_values[valid] = y_data[raster_rows[valid], raster_cols[valid]]
                valid &= self.valid_y_mask(y_values, y_nodata)

                factor_values = {}
                for name, arr in factor_arrays.items():
                    values = np.zeros(sample_r.shape, dtype="uint8")
                    in_grid = (
                        (raster_rows >= 0)
                        & (raster_rows < grid["height"])
                        & (raster_cols >= 0)
                        & (raster_cols < grid["width"])
                    )
                    values[in_grid] = arr[raster_rows[in_grid], raster_cols[in_grid]]
                    factor_values[name] = values
                    valid &= (values >= 1) & (values <= self.n_classes)

                clean_count = int(valid.sum())
                dropped += int(valid.size - clean_count)
                if clean_count:
                    ids = np.arange(kept + 1, kept + clean_count + 1)
                    block = [
                        ids,
                        sample_r[valid],
                        sample_c[valid],
                        xs[valid],
                        ys[valid],
                        y_values[valid],
                        *[factor_values[name][valid] for name in names],
                    ]
                    writer.writerows(zip(*block))
                    kept += clean_count
                self.log(f"渔网提取进度: {end:,}/{total_candidates:,}，保留 {kept:,}，剔除 {dropped:,}")

        if kept == 0:
            raise ValueError("剔除空值和异常值后，没有剩余有效采样点。")
        self.log(f"渔网提取完成，保留 {kept:,} 行，剔除 {dropped:,} 行，输出: {out_csv}")
        return out_csv, kept, dropped

    def load_clean_sample_table(self):
        sample_csv = self.sample_csv_path()
        if not sample_csv.exists():
            self.write_sample_table()
        self.log(f"读取当前间隔采样表: {sample_csv}")
        df = pd.read_csv(sample_csv)
        factor_names = [
            col for col in df.columns
            if col not in {"sample_id", "grid_row", "grid_col", "x", "y", "Y"}
        ]
        numeric_cols = ["x", "y", "Y"] + factor_names
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        valid = np.isfinite(df["Y"].to_numpy(dtype="float64"))
        valid &= np.abs(df["Y"].to_numpy(dtype="float64")) < ABNORMAL_ABS_LIMIT
        for name in factor_names:
            arr = df[name].to_numpy()
            valid &= np.isfinite(arr) & (arr >= 1) & (arr <= self.n_classes)
        before = len(df)
        df = df.loc[valid].copy()
        removed = before - len(df)
        if removed:
            self.log(f"读取采样表时再次剔除空值/异常值 {removed:,} 行。")
        if df.empty:
            raise ValueError("采样表剔除空值和异常值后为空。")
        return df, factor_names

    def run_extract(self):
        self.ensure_dirs()
        out_csv = self.sample_csv_path()
        if out_csv.exists() and self.skip_existing:
            self.log(f"跳过已有渔网提取表: {out_csv.name}")
            return out_csv, None

        self.log("开始按用户设置的间隔制作渔网，并提取 Y 与 X 值。")
        return self.write_sample_table()

    def run_detector(self):
        self.ensure_dirs()
        self.log("开始运行地理探测器。")
        sample_df, names = self.load_clean_sample_table()
        y = sample_df["Y"].to_numpy(dtype="float64")
        valid_all = np.ones(y.shape, dtype=bool)
        factor_arrays = {
            name: sample_df[name].to_numpy(dtype="uint16")
            for name in names
        }
        common_n = int(len(sample_df))
        if common_n == 0:
            raise ValueError("当前采样间隔下没有有效样点，无法运行地理探测器。")

        rows = []
        q_values = {}
        for name, arr in factor_arrays.items():
            q, n, strata = q_stat(y, arr, valid_all)
            q_values[name] = q
            rows.append({"factor": name, "q": q, "sample_count": n, "strata_count": strata})
            self.log(f"单因子 {name}: q={q:.6f}, 样点={n:,}, 分层={strata}")
        single = pd.DataFrame(rows).sort_values("q", ascending=False)

        q_matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names, dtype="float64")
        type_matrix = pd.DataFrame("", index=names, columns=names)
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    q_matrix.loc[name_i, name_j] = q_values[name_i]
                    type_matrix.loc[name_i, name_j] = "自身"
                elif j < i:
                    q_matrix.loc[name_i, name_j] = q_matrix.loc[name_j, name_i]
                    type_matrix.loc[name_i, name_j] = type_matrix.loc[name_j, name_i]
                else:
                    combo = (
                        factor_arrays[name_i].astype("uint16") * (self.n_classes + 1)
                        + factor_arrays[name_j].astype("uint16")
                    )
                    q12, _, _ = q_stat(y, combo, valid_all)
                    q_matrix.loc[name_i, name_j] = q12
                    type_matrix.loc[name_i, name_j] = interaction_type(q_values[name_i], q_values[name_j], q12)
            self.log(f"交互探测进度: {i + 1}/{len(names)}")

        single_path = self.output_dir / "单因子探测表.csv"
        q_path = self.output_dir / "交互探测q值矩阵.csv"
        type_path = self.output_dir / "交互探测类型矩阵.csv"
        summary_path = self.output_dir / "地理探测器样点摘要.txt"
        heatmap_path = self.output_dir / "交互探测热力图.png"

        single.to_csv(single_path, index=False, encoding="utf-8-sig")
        q_matrix.to_csv(q_path, encoding="utf-8-sig")
        type_matrix.to_csv(type_path, encoding="utf-8-sig")
        self.save_heatmap(q_matrix, heatmap_path)
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"采样点间隔: {self.sample_interval_m:g}m x {self.sample_interval_m:g}m\n")
            fh.write(f"采样点坐标系: {self.target_grid()['crs']}\n")
            fh.write(f"当前有效采样点数: {common_n}\n")
            fh.write(f"采样点提取表: {self.sample_csv_path()}\n")
            fh.write("X 因子: " + ", ".join(names) + "\n")

        self.log(f"地理探测器完成，输出目录: {self.output_dir}")
        return single_path, heatmap_path

    def save_heatmap(self, q_matrix, out_path):
        font = get_chinese_font()
        if font is not None:
            plt.rcParams["font.sans-serif"] = [font.get_name()]
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax = plt.subplots(figsize=(8, 6.8), dpi=220)
        matrix = q_matrix.to_numpy(dtype=float)
        upper_mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        lower_matrix = np.ma.array(matrix, mask=upper_mask)
        try:
            cmap = plt.get_cmap(self.heatmap_cmap).copy()
        except ValueError:
            self.log(f"无法识别色带 {self.heatmap_cmap}，改用 YlOrRd。")
            cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad(color="white")
        im = ax.imshow(
            lower_matrix,
            cmap=cmap,
            vmin=0,
            vmax=max(0.001, float(np.nanmax(lower_matrix))),
        )
        names = list(q_matrix.index)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontproperties=font)
        ax.set_yticklabels(names, fontproperties=font)
        ax.set_title(f"{self.y_tif.stem}地理探测器交互探测 q 值热力图", fontproperties=font, pad=14)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if j > i:
                    continue
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("q 值", rotation=270, labelpad=12)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def run_all(self):
        self.run_clip()
        self.run_classify()
        self.run_extract()
        self.run_detector()


class GeoDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("地理探测器一条龙")
        self.geometry("1040x760")
        self.minsize(940, 680)
        self.log_queue = queue.Queue()
        self.worker = None

        self.vars = {
            "factor_dir": tk.StringVar(value=""),
            "shp_path": tk.StringVar(value=""),
            "y_tif": tk.StringVar(value=""),
            "clip_dir": tk.StringVar(value=""),
            "class_dir": tk.StringVar(value=""),
            "output_dir": tk.StringVar(value=""),
            "n_classes": tk.StringVar(value="5"),
            "sample_limit": tk.StringVar(value=str(DEFAULT_JENKS_SAMPLE_LIMIT)),
            "sample_interval_m": tk.StringVar(value=str(DEFAULT_SAMPLE_INTERVAL_M)),
            "heatmap_cmap": tk.StringVar(value="黄红渐变 YlOrRd"),
            "skip_existing": tk.BooleanVar(value=True),
        }

        self.status_var = tk.StringVar(value="就绪")
        self._build_ui()
        self.after(120, self._drain_log_queue)

    def _build_ui(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("", 18, "bold"))
        style.configure("Section.TLabelframe.Label", font=("", 12, "bold"))
        style.configure("Action.TButton", padding=(14, 7))

        main = ttk.Frame(self, padding=16)
        main.pack(fill="both", expand=True)

        header = ttk.Frame(main)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="地理探测器一条龙", style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        paths = ttk.LabelFrame(main, text="路径设置", style="Section.TLabelframe")
        paths.pack(fill="x", pady=(0, 12))
        paths.columnconfigure(1, weight=1)

        self._path_row(paths, 0, "因子 tif 文件夹", "factor_dir", "dir")
        self._path_row(paths, 1, "长江流域 shp", "shp_path", "file", [("Shapefile", "*.shp")])
        self._path_row(paths, 2, "Y 栅格 tif", "y_tif", "file", [("TIF", "*.tif *.tiff")])
        self._path_row(paths, 3, "裁剪输出文件夹", "clip_dir", "dir")
        self._path_row(paths, 4, "五级分类输出文件夹", "class_dir", "dir")
        self._path_row(paths, 5, "结果输出文件夹", "output_dir", "dir")

        options = ttk.Frame(paths)
        options.grid(row=6, column=0, columnspan=3, sticky="ew", padx=10, pady=(6, 10))
        ttk.Label(options, text="分类级数").pack(side="left")
        ttk.Spinbox(options, from_=2, to=9, textvariable=self.vars["n_classes"], width=6).pack(side="left", padx=(8, 22))
        ttk.Label(options, text="Jenks 抽样上限").pack(side="left")
        ttk.Entry(options, textvariable=self.vars["sample_limit"], width=12).pack(side="left", padx=(8, 22))
        ttk.Label(options, text="采样间隔(m)").pack(side="left")
        ttk.Entry(options, textvariable=self.vars["sample_interval_m"], width=10).pack(side="left", padx=(8, 22))
        ttk.Checkbutton(options, text="已有输出则跳过", variable=self.vars["skip_existing"]).pack(side="left")

        heatmap_options = ttk.Frame(paths)
        heatmap_options.grid(row=7, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        ttk.Label(heatmap_options, text="热力图色带").pack(side="left")
        cmap_box = ttk.Combobox(
            heatmap_options,
            textvariable=self.vars["heatmap_cmap"],
            values=list(HEATMAP_CMAPS.keys()),
            width=22,
            state="readonly",
        )
        cmap_box.pack(side="left", padx=(8, 22))

        actions = ttk.LabelFrame(main, text="处理模块", style="Section.TLabelframe")
        actions.pack(fill="x", pady=(0, 12))
        for i in range(5):
            actions.columnconfigure(i, weight=1)
        buttons = [
            ("数据裁剪", self.on_clip),
            ("五级分类", self.on_classify),
            ("渔网提取 X/Y", self.on_extract),
            ("地理探测器", self.on_detector),
            ("运行全部", self.on_all),
        ]
        for i, (text, command) in enumerate(buttons):
            ttk.Button(actions, text=text, command=command, style="Action.TButton").grid(
                row=0, column=i, padx=8, pady=12, sticky="ew"
            )

        log_frame = ttk.LabelFrame(main, text="运行日志", style="Section.TLabelframe")
        log_frame.pack(fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=18, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        footer = ttk.Frame(main)
        footer.pack(fill="x", pady=(10, 0))
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="清空日志", command=self.clear_log).pack(side="left", padx=(10, 0))

    def _path_row(self, parent, row, label, var_name, mode, filetypes=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        ttk.Entry(parent, textvariable=self.vars[var_name]).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        command = lambda: self.choose_path(var_name, mode, filetypes)
        ttk.Button(parent, text="选择", command=command).grid(row=row, column=2, sticky="ew", padx=10, pady=6)

    def choose_path(self, var_name, mode, filetypes=None):
        current = self.vars[var_name].get().strip()
        initial = None
        if current:
            current_path = Path(current)
            initial_path = current_path.parent if mode == "file" else current_path
            if initial_path.exists():
                initial = str(initial_path)
        if mode == "dir":
            options = {"initialdir": initial} if initial else {}
            value = filedialog.askdirectory(**options)
        else:
            options = {"filetypes": filetypes or [("All files", "*")]}
            if initial:
                options["initialdir"] = initial
            value = filedialog.askopenfilename(**options)
        if value:
            self.vars[var_name].set(value)

    def config(self):
        cfg = {name: var.get() for name, var in self.vars.items()}
        cfg["skip_existing"] = self.vars["skip_existing"].get()
        required_paths = {
            "factor_dir": "因子 tif 文件夹",
            "shp_path": "长江流域 shp",
            "y_tif": "Y 栅格 tif",
            "clip_dir": "裁剪输出文件夹",
            "class_dir": "五级分类输出文件夹",
            "output_dir": "结果输出文件夹",
        }
        missing = [label for key, label in required_paths.items() if not str(cfg[key]).strip()]
        if missing:
            raise ValueError("请先选择：" + "、".join(missing) + "。")
        if int(cfg["n_classes"]) != 5:
            if not messagebox.askyesno("确认分类级数", "地理探测器常用五级分类。确定使用当前分类级数吗？"):
                raise RuntimeError("用户取消：分类级数不是 5。")
        if int(cfg["sample_limit"]) < 1000:
            raise ValueError("Jenks 抽样上限不能小于 1000。")
        if float(cfg["sample_interval_m"]) <= 0:
            raise ValueError("采样点间隔必须大于 0 米。")
        cfg["heatmap_cmap"] = safe_cmap_name(cfg["heatmap_cmap"])
        return cfg

    def log(self, message):
        self.log_queue.put(str(message))

    def _drain_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert("end", message + "\n")
                self.log_text.see("end")
        except queue.Empty:
            pass
        self.after(120, self._drain_log_queue)

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def run_task(self, title, task):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("任务正在运行", "当前已有任务在运行，请等待完成。")
            return
        try:
            cfg = self.config()
        except Exception as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        def wrapped():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.log("=" * 72)
            self.log(f"开始: {title}")
            try:
                processor = GeoDetectorProcessor(cfg, self.log)
                task(processor)
                self.log(f"完成: {title}")
                self.status_var.set("完成")
            except Exception:
                detail = traceback.format_exc()
                self.log(detail)
                self.status_var.set("出错")
                self.after(0, lambda: messagebox.showerror("运行出错", detail[-3000:]))
            finally:
                self.after(0, self.progress.stop)

        self.status_var.set(f"运行中: {title}")
        self.progress.start(12)
        self.worker = threading.Thread(target=wrapped, daemon=True)
        self.worker.start()

    def on_clip(self):
        self.run_task("数据裁剪", lambda p: p.run_clip())

    def on_classify(self):
        self.run_task("五级分类", lambda p: p.run_classify())

    def on_extract(self):
        self.run_task("制作渔网并提取 X/Y 值", lambda p: p.run_extract())

    def on_detector(self):
        self.run_task("运行地理探测器", lambda p: p.run_detector())

    def on_all(self):
        self.run_task("运行全部流程", lambda p: p.run_all())


def main():
    app = GeoDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
