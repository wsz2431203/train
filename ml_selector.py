import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import lightgbm as lgb
from lightgbm import LGBMRanker
from sklearn.model_selection import KFold, cross_val_score 

import xgboost as xgb
#from column_generation import ColumnGenerationSolver, VirtualFlight
# =====================
# 一、LightGBM排序模型
# =====================

def train_lightgbm_rank_model(feature_df, label_dict):
    """
    输入: 
        feature_df: compute_features() 返回的 DataFrame（每行一个飞机特征）
        label_dict: {tail: rank_label}
    输出:
        已训练的 LightGBM 排序模型
    """
    X = feature_df.drop(columns=["tail"]).fillna(0).values
    y = np.array([ -label_dict.get(t, 1) for t in feature_df["tail"] ])  # LightGBM中越大越重要
    group = [len(y)]

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200
    )
    model.fit(X, y, group=group)
    print("✅ LightGBM 排序模型训练完成")
    return model, feature_df.drop(columns=["tail"]).columns.tolist()



# ======================================
# 二、特征计算函数（表 B.1）
# ======================================

def compute_features(df, disrupted_flight_id, ad, candidate_ac_list):
    """计算表 B.1 的27项特征"""
    features = []

    # --- 提取目标航班 fd 和 vfd ---
    fd = df[df["flight_id"] == disrupted_flight_id].iloc[0]
    ad_df = df[df["tail"] == ad]
    window_start = ad_df["dep_time"].min()
    window_end = ad_df["arr_time"].max()
    window_len = (window_end - window_start).total_seconds() / 3600.0

    # 航班密度统计
    all_flights = df[df["tail"].isin(candidate_ac_list)]

    # --- 公共统计 ---
    for tail in candidate_ac_list:
        ac_df = df[df["tail"] == tail]

        # 基本信息特征
        CAN = len(candidate_ac_list) - 1
        NDF = len(ad_df)
        NCF_SUM = len(all_flights)
        NCF_AVE = len(all_flights) / len(candidate_ac_list)
        STWS = window_start.timestamp()
        STWE = window_end.timestamp()
        TWL = window_len
        DT = int(fd.get("disrupt_type", 0))  # 假设有字段

        # 与目标航班 fd 相关的特征
        DT_DF = fd["dep_time"].timestamp()
        AT_DF = fd["arr_time"].timestamp()
        DFT = (fd["arr_time"] - fd["dep_time"]).total_seconds() / 3600.0
        DT_DF_vs_STWS = DT_DF - STWS
        AT_DF_vs_STWE = AT_DF - STWE

        # 选取 ad 的下一个航班 vfd
        ad_future = ad_df[ad_df["dep_time"] > fd["arr_time"]]
        if not ad_future.empty:
            vf = ad_future.iloc[0]
            DT_VF = vf["dep_time"].timestamp()
            AT_VF = vf["arr_time"].timestamp()
            VFT = (vf["arr_time"] - vf["dep_time"]).total_seconds() / 3600.0
        else:
            DT_VF = AT_VF = VFT = 0
        DT_VF_vs_STWS = DT_VF - STWS
        AT_VF_vs_STWE = AT_VF - STWE

        # 计算地面停留时间
        ac_df = ac_df.sort_values("dep_time")
        turn_times = []
        for i in range(1, len(ac_df)):
            delta = (ac_df.iloc[i]["dep_time"] - ac_df.iloc[i - 1]["arr_time"]).total_seconds() / 3600.0
            turn_times.append(max(delta, 0))
        DST = np.sum(turn_times)

        # 所有飞机地面停留时间分布
        all_turns = []
        for _, group in all_flights.groupby("tail"):
            group = group.sort_values("dep_time")
            for i in range(1, len(group)):
                delta = (group.iloc[i]["dep_time"] - group.iloc[i - 1]["arr_time"]).total_seconds() / 3600.0
                all_turns.append(max(delta, 0))
        CST_MIN = np.min(all_turns) if all_turns else 0
        CST_MAX = np.max(all_turns) if all_turns else 0
        CST_AVE = np.mean(all_turns) if all_turns else 0

        # 所有航班飞行时间分布
        all_durations = (all_flights["arr_time"] - all_flights["dep_time"]).dt.total_seconds() / 3600.0
        SD_CF_MIN = all_durations.min()
        SD_CF_MAX = all_durations.max()
        SD_CF_AVE = all_durations.mean()

        # 在 fd 出发前1小时出发机场的飞机数
        CNA_1HB = np.sum(
            (all_flights["arr_airport"] == fd["dep_airport"]) &
            (abs((fd["dep_time"] - all_flights["arr_time"]).dt.total_seconds()) <= 3600)
        )
        # 在 fd 到达后1小时到达机场的飞机数
        CNA_1HF = np.sum(
            (all_flights["dep_airport"] == fd["arr_airport"]) &
            (abs((fd["arr_time"] - all_flights["dep_time"]).dt.total_seconds()) <= 3600)
        )

        features.append({
            "tail": tail,
            "CAN": CAN, "NDF": NDF, "NCF_SUM": NCF_SUM, "NCF_AVE": NCF_AVE,
            "STWS": STWS, "STWE": STWE, "TWL": TWL, "DT": DT,
            "DT_DF": DT_DF, "AT_DF": AT_DF, "DFT": DFT,
            "DT_DF_vs_STWS": DT_DF_vs_STWS, "AT_DF_vs_STWE": AT_DF_vs_STWE,
            "DT_VF": DT_VF, "AT_VF": AT_VF, "VFT": VFT,
            "DT_VF_vs_STWS": DT_VF_vs_STWS, "AT_VF_vs_STWE": AT_VF_vs_STWE,
            "DST": DST, "CST_MIN": CST_MIN, "CST_MAX": CST_MAX, "CST_AVE": CST_AVE,
            "SD_CF_MIN": SD_CF_MIN, "SD_CF_MAX": SD_CF_MAX, "SD_CF_AVE": SD_CF_AVE,
            "CNA_1HB": CNA_1HB, "CNA_1HF": CNA_1HF
        })

    return pd.DataFrame(features)


# ======================================
# 三、机器学习飞机选择器 (排序+数量预测)
# ======================================

class MLAircraftSelector:
    def __init__(self, rank_model=None, feature_names=None):
        self.ranker_model = rank_model
        self.feature_names = feature_names
        self.count_model = None

    def predict_aircraft_ranking(self, feature_df):
        X = feature_df[self.feature_names].fillna(0).values
        preds = self.ranker_model.predict(X)
        feature_df["score"] = preds
        ranked = feature_df.sort_values("score", ascending=False)
        return [(r.tail, r.score) for r in ranked.itertuples()]

    def _solve_with_selected_aircraft(self, case: Dict, selected_aircraft_ids: List[str]) -> Tuple[float, Dict]:
        solver = ColumnGenerationSolver()
        for fid, f in case.get('flights', {}).items(): solver.add_flight(f)
        for aid, ac in case.get('aircraft', {}).items():
            if aid in selected_aircraft_ids: solver.add_aircraft(ac)
        for mid, m in case.get('maintenances', {}).items(): solver.add_maintenance(m)
        for sid, s in case.get('slots', {}).items(): solver.add_slot(s)
        result = solver.solve(disrupted_flight_ids=[case['disrupted_flight_id']], use_ml_selection=False)
        total_cost = sum(p.get("cost", 1000) for p in result.get("paths", [])) or 2000
        return total_cost, result

    def derive_optimal_count_by_search(self, case: Dict, feature_df: pd.DataFrame):
        ranking = self.predict_aircraft_ranking(feature_df)
        ranked_ids = [aid for aid, _ in ranking]
        best_k, best_cost = 1, float("inf")
        for k in range(1, len(ranked_ids) + 1):
            selected = ranked_ids[:k]
            cost_k, _ = self._solve_with_selected_aircraft(case, selected)
            if cost_k < best_cost:
                best_cost, best_k = cost_k, k
        return best_k

    def train_count_predictor(self, training_cases: List[Dict], feature_dfs: List[pd.DataFrame]):
        X_list, y_list = [], []
        for case, feat_df in zip(training_cases, feature_dfs):
            optimal_k = self.derive_optimal_count_by_search(case, feat_df)
            case["optimal_aircraft_count"] = optimal_k
            X_list.append([feat_df.shape[0], feat_df["score"].mean(), feat_df["score"].std()])
            y_list.append(optimal_k)

        X, y = np.array(X_list), np.array(y_list)
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            objective='reg:squarederror', random_state=42
        )
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
        print(f"📘 回归模型 5-fold CV MSE = {-cv_scores.mean():.3f}")
        model.fit(X, y)
        self.count_model = model
        print("✅ 飞机数量预测模型训练完成")
        return model



