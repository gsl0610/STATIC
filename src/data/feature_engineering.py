"""
数据准备模块 - 特征工程与样本构建

负责将 DSP 广告原始日志转换为模型可用的训练样本，包括：
1. 特征编码（类别特征 → ID映射）
2. 用户行为序列构建
3. 多任务标签处理（CTR / CVR）
4. 数据集划分与持久化
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """类别特征 → 整数ID 的统一编码器。

    支持：
    - 单值特征（gender, city_level 等）
    - 多值特征（interest_tags 等，逗号分隔）
    - 未知值映射到 OOV ID (index=1), padding 为 0
    """

    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.vocab: dict[str, dict[str, int]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: list[str],
            multi_value_cols: list[str] | None = None) -> FeatureEncoder:
        """统计词频并构建词表。"""
        multi_value_cols = set(multi_value_cols or [])

        for col in feature_cols:
            counter: dict[str, int] = defaultdict(int)
            for val in df[col].dropna().astype(str):
                if col in multi_value_cols:
                    for v in val.split(","):
                        v = v.strip()
                        if v:
                            counter[v] += 1
                else:
                    counter[val] += 1

            # 0=PAD, 1=OOV, 2+ = 有效值
            mapping: dict[str, int] = {}
            idx = 2
            for k, cnt in sorted(counter.items(), key=lambda x: -x[1]):
                if cnt >= self.min_freq:
                    mapping[k] = idx
                    idx += 1
            self.vocab[col] = mapping
            logger.info(f"特征 [{col}] 词表大小: {len(mapping) + 2} (含PAD+OOV)")

        self._fitted = True
        return self

    def transform_single(self, col: str, value: str) -> int:
        """单值特征编码。"""
        return self.vocab.get(col, {}).get(str(value), 1)

    def transform_multi(self, col: str, value: str, max_len: int) -> list[int]:
        """多值特征编码，截断/填充到 max_len。"""
        mapping = self.vocab.get(col, {})
        ids = []
        for v in str(value).split(","):
            v = v.strip()
            if v:
                ids.append(mapping.get(v, 1))
        # 截断 + 填充
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    def get_vocab_size(self, col: str) -> int:
        return len(self.vocab.get(col, {})) + 2

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"特征编码器已保存到 {path}")

    def load(self, path: str) -> FeatureEncoder:
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self._fitted = True
        logger.info(f"特征编码器已加载: {path}")
        return self


class BehaviorSequenceBuilder:
    """用户行为序列构建器。

    将用户历史点击/转化广告列表转换为定长序列。
    """

    def __init__(self, max_seq_len: int = 50):
        self.max_seq_len = max_seq_len

    def build(self, ad_id_list: list[int]) -> np.ndarray:
        """将广告ID列表转为定长序列（左截断，右填充0）。"""
        seq = ad_id_list[-self.max_seq_len:]
        padded = [0] * (self.max_seq_len - len(seq)) + seq
        return np.array(padded, dtype=np.int64)


class DSPDatasetBuilder:
    """DSP 广告数据集构建器。

    从原始日志 DataFrame 构建训练/验证/测试集。
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        data_cfg = config["data"]

        self.user_features = data_cfg["user_features"]
        self.ad_features = data_cfg["ad_features"]
        self.context_features = data_cfg["context_features"]
        self.stat_features = data_cfg.get("stat_features", [])
        self.label_cols = data_cfg["labels"]

        # 多值特征自动检测: interest_tags, interest_tags_l2, creative_label
        self.multi_value_cols = [
            col for col in (self.user_features + self.ad_features)
            if col in ("interest_tags", "interest_tags_l2", "creative_label")
        ]

        self.encoder = FeatureEncoder(min_freq=5)
        self.seq_builder = BehaviorSequenceBuilder(
            max_seq_len=data_cfg["behavior"]["max_seq_len"]
        )
        self.behavior_col = data_cfg["behavior"]["feature_name"]

    def build_from_dataframe(self, df: pd.DataFrame,
                             output_dir: str) -> dict[str, str]:
        """端到端数据构建流程。

        Args:
            df: 原始日志 DataFrame，列名需与配置中定义的特征名一致。
            output_dir: 输出目录。

        Returns:
            各数据集文件路径字典。
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. 特征编码 (包含统计特征, 均作为分桶类别处理)
        all_cat_cols = (self.user_features + self.ad_features
                        + self.context_features + self.stat_features)
        self.encoder.fit(df, all_cat_cols, self.multi_value_cols)
        self.encoder.save(os.path.join(output_dir, "feature_encoder.json"))

        # 2. 编码所有特征
        encoded_data = self._encode_features(df, all_cat_cols, self.multi_value_cols)

        # 3. 构建行为序列
        if self.behavior_col in df.columns:
            behavior_seqs = self._build_behavior_sequences(df)
            encoded_data["behavior_seq"] = behavior_seqs

        # 4. 标签
        for label_col in self.label_cols:
            if label_col in df.columns:
                encoded_data[label_col] = df[label_col].values.astype(np.float32)

        # 5. 数据划分
        split_cfg = self.config["data"]["split"]
        splits = self._split_data(df, encoded_data, split_cfg)

        # 6. 保存
        paths = {}
        for split_name, split_data in splits.items():
            path = os.path.join(output_dir, f"{split_name}.npz")
            np.savez_compressed(path, **split_data)
            paths[split_name] = path
            logger.info(f"[{split_name}] 样本数: {len(split_data[self.label_cols[0]])}, 保存到: {path}")

        return paths

    def _encode_features(self, df: pd.DataFrame,
                         cat_cols: list[str],
                         multi_value_cols: list[str]) -> dict[str, np.ndarray]:
        """批量编码所有类别特征。"""
        result = {}
        multi_set = set(multi_value_cols)

        for col in cat_cols:
            if col in multi_set:
                max_len = 10  # 多值特征最大长度
                encoded = np.array([
                    self.encoder.transform_multi(col, str(v), max_len)
                    for v in df[col].fillna("").values
                ], dtype=np.int64)
                result[col] = encoded
            else:
                encoded = np.array([
                    self.encoder.transform_single(col, str(v))
                    for v in df[col].fillna("").values
                ], dtype=np.int64)
                result[col] = encoded

        return result

    def _build_behavior_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """构建用户行为序列。"""
        seqs = []
        for val in df[self.behavior_col].fillna("").values:
            if isinstance(val, str) and val:
                ad_ids = [int(x) for x in val.split(",") if x.strip()]
            elif isinstance(val, (list, np.ndarray)):
                ad_ids = [int(x) for x in val]
            else:
                ad_ids = []
            seqs.append(self.seq_builder.build(ad_ids))
        return np.stack(seqs)

    def _split_data(self, df: pd.DataFrame,
                    encoded_data: dict[str, np.ndarray],
                    split_cfg: dict) -> dict[str, dict[str, np.ndarray]]:
        """按日期或随机划分数据。"""
        n = len(df)
        split_by = split_cfg.get("split_by", "random")

        if split_by == "date" and "date" in df.columns:
            dates = pd.to_datetime(df["date"])
            sorted_dates = dates.sort_values().unique()
            n_dates = len(sorted_dates)
            train_end = sorted_dates[int(n_dates * split_cfg["train_ratio"])]
            val_end = sorted_dates[int(n_dates * (split_cfg["train_ratio"] + split_cfg["val_ratio"]))]

            train_mask = dates <= train_end
            val_mask = (dates > train_end) & (dates <= val_end)
            test_mask = dates > val_end
        else:
            indices = np.random.permutation(n)
            train_size = int(n * split_cfg["train_ratio"])
            val_size = int(n * split_cfg["val_ratio"])

            train_mask = np.zeros(n, dtype=bool)
            val_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size + val_size]] = True
            test_mask[indices[train_size + val_size:]] = True

        splits = {}
        for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            split_data = {}
            for key, arr in encoded_data.items():
                split_data[key] = arr[mask] if isinstance(mask, np.ndarray) else arr[mask.values]
            splits[name] = split_data

        return splits


def generate_synthetic_data(num_samples: int = 100000) -> pd.DataFrame:
    """生成接近真实的 DSP 广告日志数据 (参考腾讯AMS/字节巨量引擎/快手磁力/美团广告)。

    特征体系覆盖:
      - 用户画像: 人口属性 + 设备环境 + 消费能力 + 兴趣标签 + 活跃度
      - 广告属性: 行业类目 + 创意维度 + 出价设置 + 投放状态
      - 上下文: 时间 + 媒体 + 广告位 + 请求环境 + 竞争信号
      - 统计特征: 实时CTR/CVR + 用户侧统计 + 交叉统计
      - 行为序列: 用户历史点击广告ID序列
    """
    rng = np.random.RandomState(42)

    # ======================== 用户特征 ========================
    # --- 基础人口属性 ---
    user_ids = [f"u_{i}" for i in rng.randint(0, 80000, num_samples)]
    age_buckets = rng.choice(
        ["18-24", "25-34", "35-44", "45-54", "55+"],
        num_samples, p=[0.22, 0.35, 0.22, 0.13, 0.08]
    )
    genders = rng.choice(["M", "F", "U"], num_samples, p=[0.46, 0.44, 0.10])
    city_levels = rng.choice(
        ["tier1", "new_tier1", "tier2", "tier3", "tier4", "tier5_below"],
        num_samples, p=[0.12, 0.15, 0.20, 0.22, 0.18, 0.13]
    )
    provinces = rng.choice([
        "guangdong", "zhejiang", "jiangsu", "beijing", "shanghai",
        "shandong", "henan", "sichuan", "hubei", "hunan",
        "fujian", "anhui", "hebei", "liaoning", "shaanxi",
        "chongqing", "jiangxi", "guangxi", "yunnan", "tianjin",
        "heilongjiang", "jilin", "shanxi", "guizhou", "gansu",
        "inner_mongolia", "xinjiang", "hainan", "ningxia", "tibet",
        "qinghai", "other"
    ], num_samples)
    life_stages = rng.choice(
        ["student", "young_worker", "married_no_child", "married_with_child", "retired"],
        num_samples, p=[0.15, 0.30, 0.15, 0.30, 0.10]
    )

    # --- 设备环境 (参考快手/字节设备分布) ---
    device_types = rng.choice(
        ["phone", "tablet", "pc", "smart_tv"],
        num_samples, p=[0.78, 0.06, 0.14, 0.02]
    )
    os_types = rng.choice(
        ["android", "ios", "windows", "mac", "harmonyos", "linux"],
        num_samples, p=[0.52, 0.28, 0.10, 0.04, 0.05, 0.01]
    )
    device_brands = rng.choice(
        ["apple", "huawei", "xiaomi", "oppo", "vivo", "samsung",
         "honor", "realme", "lenovo", "oneplus", "meizu", "other"],
        num_samples, p=[0.25, 0.15, 0.13, 0.10, 0.10, 0.05,
                        0.06, 0.03, 0.03, 0.02, 0.02, 0.06]
    )
    device_price_levels = rng.choice(
        ["low", "mid", "high", "premium"],
        num_samples, p=[0.20, 0.35, 0.30, 0.15]
    )
    network_types = rng.choice(
        ["wifi", "4g", "5g", "3g", "2g"],
        num_samples, p=[0.45, 0.28, 0.22, 0.04, 0.01]
    )
    carriers = rng.choice(
        ["cmcc", "cucc", "ctcc", "other"],
        num_samples, p=[0.45, 0.25, 0.25, 0.05]
    )
    screen_resolutions = rng.choice(
        ["720p", "1080p", "2k", "4k"],
        num_samples, p=[0.15, 0.50, 0.30, 0.05]
    )

    # --- 消费能力 (参考腾讯AMS画像) ---
    consumption_levels = rng.choice(
        ["low", "medium", "high", "premium"],
        num_samples, p=[0.20, 0.40, 0.28, 0.12]
    )
    pay_channel_prefs = rng.choice(
        ["wechat_pay", "alipay", "bank_card", "credit"],
        num_samples, p=[0.40, 0.35, 0.15, 0.10]
    )

    # --- 兴趣标签 (参考字节巨量引擎多级兴趣标签体系) ---
    interest_l1_pool = [
        "sports", "tech", "fashion", "food", "travel", "game", "finance",
        "education", "health", "entertainment", "auto", "real_estate",
        "parenting", "beauty", "home_decor", "pet", "reading", "music",
        "photography", "outdoor"
    ]
    interest_l2_pool = [
        "basketball", "football", "running", "yoga", "smartphone", "laptop",
        "ai_tech", "iot", "streetwear", "luxury", "skincare", "makeup",
        "chinese_food", "western_food", "baking", "coffee", "domestic_travel",
        "overseas_travel", "moba_game", "fps_game", "puzzle_game",
        "stock", "fund", "insurance", "k12", "language", "cert_exam",
        "fitness", "nutrition", "diet", "movie", "variety_show", "drama",
        "suv", "sedan", "ev_car", "new_house", "rental", "infant",
        "toddler", "cat", "dog", "novel", "comic", "pop_music",
        "classical", "landscape_photo", "portrait_photo", "hiking", "camping"
    ]
    interest_tags = [
        ",".join(rng.choice(interest_l1_pool, size=rng.randint(1, 7), replace=False))
        for _ in range(num_samples)
    ]
    interest_tags_l2 = [
        ",".join(rng.choice(interest_l2_pool, size=rng.randint(1, 10), replace=False))
        for _ in range(num_samples)
    ]

    # --- 活跃度 ---
    user_active_levels = rng.choice(
        ["dormant", "low", "medium", "high", "super"],
        num_samples, p=[0.08, 0.20, 0.35, 0.27, 0.10]
    )
    reg_days_buckets = rng.choice(
        ["new_7d", "30d", "90d", "180d", "365d", "veteran"],
        num_samples, p=[0.05, 0.08, 0.12, 0.15, 0.25, 0.35]
    )

    # ======================== 广告特征 ========================
    ad_ids = [f"ad_{i}" for i in rng.randint(0, 50000, num_samples)]
    campaign_ids = [f"camp_{i}" for i in rng.randint(0, 8000, num_samples)]
    advertiser_ids = [f"adv_{i}" for i in rng.randint(0, 3000, num_samples)]
    ad_account_ids = [f"acc_{i}" for i in rng.randint(0, 5000, num_samples)]

    ad_categories = rng.choice(
        ["ecommerce", "app_install", "lead_gen", "brand", "local_service", "game"],
        num_samples, p=[0.30, 0.20, 0.15, 0.12, 0.13, 0.10]
    )
    ad_industry_l1 = rng.choice([
        "ecommerce", "game", "education", "finance", "auto", "real_estate",
        "local_life", "healthcare", "fmcg", "it_service", "travel",
        "social_dating", "tool_app", "media_entertainment"
    ], num_samples)
    ad_industry_l2 = rng.choice([
        "clothing", "3c_digital", "beauty_cosmetic", "food_beverage",
        "household", "moba", "slg", "casual_game", "k12_edu",
        "adult_edu", "vocational", "bank", "insurance", "securities",
        "new_energy_car", "luxury_car", "used_car", "commercial_housing",
        "decoration", "catering", "hotel", "local_service", "pharmacy",
        "dental", "medical_beauty", "dairy", "snack", "personal_care",
        "cloud_service", "saas", "hotel_booking", "flight_ticket",
        "social_app", "dating_app", "photo_tool", "fitness_app",
        "short_video", "live_stream", "reading_app"
    ], num_samples)

    creative_types = rng.choice(
        ["single_image", "multi_image", "horizontal_video", "vertical_video",
         "carousel", "live_feed", "interactive"],
        num_samples, p=[0.20, 0.10, 0.15, 0.25, 0.10, 0.12, 0.08]
    )
    creative_sizes = rng.choice(
        ["640x100", "720x1280", "1080x1920", "1280x720", "750x1334",
         "1242x2208", "300x250", "320x50"],
        num_samples
    )
    creative_duration_buckets = rng.choice(
        ["0s", "0-6s", "6-15s", "15-30s", "30-60s", "60s+"],
        num_samples, p=[0.30, 0.15, 0.25, 0.15, 0.10, 0.05]
    )
    has_cta_buttons = rng.choice(["yes", "no"], num_samples, p=[0.65, 0.35])

    creative_label_pool = [
        "promotion", "celebrity", "storyline", "voiceover", "review",
        "tutorial", "unboxing", "comparison", "testimonial", "countdown",
        "festival", "limited_offer", "free_trial"
    ]
    creative_labels = [
        ",".join(rng.choice(creative_label_pool, size=rng.randint(1, 4), replace=False))
        for _ in range(num_samples)
    ]

    landing_page_types = rng.choice(
        ["app_store", "h5", "deeplink", "mini_program", "live_room", "shop", "form"],
        num_samples, p=[0.18, 0.22, 0.15, 0.15, 0.10, 0.12, 0.08]
    )
    bid_types = rng.choice(
        ["cpc", "cpm", "ocpc", "ocpm", "cpa"],
        num_samples, p=[0.10, 0.08, 0.30, 0.40, 0.12]
    )
    delivery_speeds = rng.choice(
        ["standard", "accelerated"],
        num_samples, p=[0.70, 0.30]
    )
    ad_statuses = rng.choice(
        ["active", "paused", "budget_exhausted", "pending_review"],
        num_samples, p=[0.70, 0.10, 0.12, 0.08]
    )

    # ======================== 上下文特征 ========================
    hours = rng.randint(0, 24, num_samples)
    days = rng.randint(0, 7, num_samples)
    is_weekends = np.where(days >= 5, "yes", "no")
    time_periods = np.array([
        "midnight" if h < 5 else "dawn" if h < 7 else "morning" if h < 11
        else "noon" if h < 14 else "afternoon" if h < 17
        else "evening" if h < 21 else "night"
        for h in hours
    ])
    is_holidays = rng.choice(["yes", "no"], num_samples, p=[0.08, 0.92])

    media_ids = [f"media_{i}" for i in rng.randint(0, 200, num_samples)]
    media_types = rng.choice(
        ["social", "video", "news", "search", "ecommerce", "lifestyle"],
        num_samples, p=[0.25, 0.25, 0.15, 0.12, 0.13, 0.10]
    )
    slot_ids = [f"slot_{i}" for i in rng.randint(0, 1000, num_samples)]
    slot_types = rng.choice(
        ["feed_stream", "banner", "interstitial", "rewarded", "splash", "search", "pre_roll"],
        num_samples, p=[0.35, 0.15, 0.10, 0.10, 0.08, 0.12, 0.10]
    )
    slot_positions = rng.choice(
        ["1st", "2nd", "3rd", "4th", "5th+"],
        num_samples, p=[0.25, 0.22, 0.20, 0.18, 0.15]
    )
    request_types = rng.choice(
        ["organic", "refresh", "loadmore", "push"],
        num_samples, p=[0.40, 0.30, 0.20, 0.10]
    )
    page_categories = rng.choice(
        ["home", "detail", "search", "cart", "profile", "video"],
        num_samples, p=[0.30, 0.20, 0.15, 0.10, 0.10, 0.15]
    )
    competing_ads_buckets = rng.choice(
        ["0-5", "5-10", "10-20", "20-50", "50+"],
        num_samples, p=[0.10, 0.25, 0.35, 0.20, 0.10]
    )
    ecpm_buckets = rng.choice(
        ["low", "medium", "high", "premium"],
        num_samples, p=[0.25, 0.35, 0.25, 0.15]
    )

    # ======================== 统计特征 (分桶) ========================
    stat_bucket = lambda low, high: rng.choice(
        [f"b{i}" for i in range(8)], num_samples
    )
    ad_ctr_1h = stat_bucket(0, 0.2)
    ad_ctr_24h = stat_bucket(0, 0.15)
    ad_cvr_24h = stat_bucket(0, 0.05)
    ad_imp_cnt_1h = stat_bucket(0, 10000)
    ad_click_cnt_24h = stat_bucket(0, 5000)
    user_imp_cnt_24h = stat_bucket(0, 200)
    user_click_cnt_24h = stat_bucket(0, 50)
    user_category_ctr = stat_bucket(0, 0.3)
    user_x_ad_category_imp = stat_bucket(0, 100)
    user_x_advertiser_click = stat_bucket(0, 50)
    user_x_industry_cvr = stat_bucket(0, 0.1)

    # ======================== 行为序列 ========================
    click_ad_seqs = [
        ",".join([str(x) for x in rng.randint(0, 50000, rng.randint(0, 40))])
        for _ in range(num_samples)
    ]

    # ======================== 日期 ========================
    dates = pd.date_range("2025-01-01", periods=num_samples, freq="min") \
              .strftime("%Y-%m-%d").tolist()

    # ======================== 组装 DataFrame ========================
    data = {
        # 用户特征
        "user_id": user_ids,
        "age_bucket": age_buckets,
        "gender": genders,
        "city_level": city_levels,
        "province": provinces,
        "life_stage": life_stages,
        "device_type": device_types,
        "os_type": os_types,
        "device_brand": device_brands,
        "device_price_level": device_price_levels,
        "network_type": network_types,
        "carrier": carriers,
        "screen_resolution": screen_resolutions,
        "consumption_level": consumption_levels,
        "pay_channel_pref": pay_channel_prefs,
        "interest_tags": interest_tags,
        "interest_tags_l2": interest_tags_l2,
        "user_active_level": user_active_levels,
        "reg_days_bucket": reg_days_buckets,
        # 广告特征
        "ad_id": ad_ids,
        "campaign_id": campaign_ids,
        "advertiser_id": advertiser_ids,
        "ad_account_id": ad_account_ids,
        "ad_category": ad_categories,
        "ad_industry_l1": ad_industry_l1,
        "ad_industry_l2": ad_industry_l2,
        "creative_type": creative_types,
        "creative_size": creative_sizes,
        "creative_duration_bucket": creative_duration_buckets,
        "has_cta_button": has_cta_buttons,
        "creative_label": creative_labels,
        "landing_page_type": landing_page_types,
        "bid_type": bid_types,
        "delivery_speed": delivery_speeds,
        "ad_status": ad_statuses,
        # 上下文特征
        "hour_of_day": hours,
        "day_of_week": days,
        "is_weekend": is_weekends,
        "time_period": time_periods,
        "is_holiday": is_holidays,
        "media_id": media_ids,
        "media_type": media_types,
        "slot_id": slot_ids,
        "slot_type": slot_types,
        "slot_position": slot_positions,
        "request_type": request_types,
        "page_category": page_categories,
        "competing_ads_bucket": competing_ads_buckets,
        "ecpm_bucket": ecpm_buckets,
        # 统计特征
        "ad_ctr_1h": ad_ctr_1h,
        "ad_ctr_24h": ad_ctr_24h,
        "ad_cvr_24h": ad_cvr_24h,
        "ad_imp_cnt_1h": ad_imp_cnt_1h,
        "ad_click_cnt_24h": ad_click_cnt_24h,
        "user_imp_cnt_24h": user_imp_cnt_24h,
        "user_click_cnt_24h": user_click_cnt_24h,
        "user_category_ctr": user_category_ctr,
        "user_x_ad_category_imp": user_x_ad_category_imp,
        "user_x_advertiser_click": user_x_advertiser_click,
        "user_x_industry_cvr": user_x_industry_cvr,
        # 行为序列
        "click_ad_seq": click_ad_seqs,
        # 日期
        "date": dates,
    }

    # ======================== 标签生成 (模拟真实CTR分布) ========================
    # 真实广告CTR通常在 2%-8%，受广告位/创意/用户匹配度影响
    # CVR 通常在 0.5%-3%，是点击后转化
    base_ctr = 0.04
    # 广告位加成
    slot_ctr_boost = np.where(np.isin(slot_types, ["feed_stream", "search"]), 0.02,
                     np.where(np.isin(slot_types, ["rewarded", "splash"]), 0.01, 0.0))
    # 位置衰减
    pos_decay = np.where(np.array(slot_positions) == "1st", 0.02,
                np.where(np.array(slot_positions) == "2nd", 0.01, 0.0))
    # 创意加成 (视频 > 图片)
    creative_boost = np.where(
        np.isin(creative_types, ["vertical_video", "horizontal_video", "live_feed"]), 0.015, 0.0
    )
    ctr_prob = np.clip(base_ctr + slot_ctr_boost + pos_decay + creative_boost, 0.01, 0.15)
    click = rng.binomial(1, ctr_prob).astype(np.float32)

    # CVR: 点击后 15-25% 转化 (取决于落地页类型)
    base_cvr = 0.18
    lp_boost = np.where(np.isin(landing_page_types, ["deeplink", "mini_program"]), 0.05,
               np.where(np.array(landing_page_types) == "form", 0.03, 0.0))
    cvr_prob = np.clip(base_cvr + lp_boost, 0.08, 0.35)
    conversion = np.where(click == 1, rng.binomial(1, cvr_prob), 0).astype(np.float32)

    data["click"] = click
    data["conversion"] = conversion

    df = pd.DataFrame(data)
    logger.info(
        f"生成模拟数据: {num_samples} 条 | "
        f"特征维度: 用户{19} + 广告{16} + 上下文{14} + 统计{11} = 60个特征 | "
        f"CTR={click.mean():.4f}, CVR={conversion.mean():.4f}"
    )
    return df
