"""
特征提取模块
用于从问题数据中提取机器学习所需的特征
"""
import numpy as np
from typing import Dict, List


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, problem_data: Dict):
        """
        初始化特征提取器
        
        Args:
            problem_data: 问题数据字典，包含aircraft、flights等信息
        """
        self.problem_data = problem_data

    # ===============================================================
    # 一、排序模型特征（65维）
    # ===============================================================
    def extract_aircraft_features(self, aircraft_id: str, disrupted_flight_id: str) -> np.ndarray:
        """
        提取飞机排序模型特征（65维）

        Args:
            aircraft_id: 飞机ID
            disrupted_flight_id: 受干扰航班ID
        Returns:
            65维特征向量
        """
        features = []
        aircraft = self.problem_data['aircraft'][aircraft_id]
        disrupted_flight = self.problem_data['flights'][disrupted_flight_id]

        # ---------------------------
        # (1) 航班网络基本信息特征 1–8
        # ---------------------------
        all_aircraft = self.problem_data['aircraft']
        all_flights = self.problem_data['flights']
        time_window = self.problem_data.get("time_window", 24.0)
        time_window_start = min(f["dep_time"] for f in all_flights.values())
        time_window_end = max(f["arr_time"] for f in all_flights.values())

        CAN = len(all_aircraft) - 1
        NDF = len(aircraft["planned_flights"])
        NCF_SUM = sum(len(ac["planned_flights"]) for ac in all_aircraft.values())
        NCF_AVE = NCF_SUM / len(all_aircraft)
        STWS = time_window_start
        STWE = time_window_end
        TWL = STWE - STWS
        DT_type = disrupted_flight.get("disruption_type", "turn_time")

        # 特征 1–7
        features += [CAN, NDF, NCF_SUM, NCF_AVE, STWS, STWE, TWL]
        # 特征 8 one-hot 编码
        for dtype in ["turn_time", "airport_conflict", "airport_closure"]:
            features.append(1 if DT_type == dtype else 0)

        # ---------------------------
        # (2) 与目标不正常航班相关特征 9–19
        # ---------------------------
        DT_DF = disrupted_flight["dep_time"]
        AT_DF = disrupted_flight["arr_time"]
        DFT = AT_DF - DT_DF
        DT_DF_vs_STWS = DT_DF - STWS
        AT_DF_vs_STWE = AT_DF - STWE

        # 虚拟的替代航班（vf）
        related_flights = [self.problem_data['flights'][fid] for fid in aircraft['planned_flights']] if aircraft['planned_flights'] else []
        if related_flights:
            vf = related_flights[0]
            DT_VF = vf["dep_time"]
            AT_VF = vf["arr_time"]
            VFT = AT_VF - DT_VF
            DT_VF_vs_STWS = DT_VF - STWS
            AT_VF_vs_STWE = AT_VF - STWE
        else:
            DT_VF = AT_VF = VFT = DT_VF_vs_STWS = AT_VF_vs_STWE = 0

        DST = self._calculate_total_ground_time(aircraft)

        features += [
            DT_DF, AT_DF, DFT,
            DT_DF_vs_STWS, AT_DF_vs_STWE,
            DT_VF, AT_VF, VFT,
            DT_VF_vs_STWS, AT_VF_vs_STWE,
            DST
        ]

        # ---------------------------
        # (3) 航班密度特征 20–25
        # ---------------------------
        ground_times = [self._calculate_total_ground_time(ac) for ac in all_aircraft.values()]
        flight_times = [f["arr_time"] - f["dep_time"] for f in all_flights.values()]

        CST_MIN, CST_MAX, CST_AVE = min(ground_times), max(ground_times), np.mean(ground_times)
        SD_CF_MIN, SD_CF_MAX, SD_CF_AVE = min(flight_times), max(flight_times), np.mean(flight_times)

        features += [CST_MIN, CST_MAX, CST_AVE, SD_CF_MIN, SD_CF_MAX, SD_CF_AVE]

        # ---------------------------
        # (4) 其他机场密度特征 26–27
        # ---------------------------
        dep_airport = disrupted_flight["dep_airport"]
        arr_airport = disrupted_flight["arr_airport"]
        DT_DF_time = disrupted_flight["dep_time"]
        AT_DF_time = disrupted_flight["arr_time"]

        CNA_1HB = sum(
            1 for f in all_flights.values()
            if f["arr_airport"] == dep_airport and 0 <= (DT_DF_time - f["arr_time"]) <= 1
        )
        CNA_1HF = sum(
            1 for f in all_flights.values()
            if f["dep_airport"] == arr_airport and 0 <= (f["dep_time"] - AT_DF_time) <= 1
        )
        features += [CNA_1HB, CNA_1HF]

        # ---------------------------
        # (5) 其他补充飞机特征 28–65（飞行性能+机场匹配性等）
        # ---------------------------
        # 飞机基本性能
        features.append(aircraft.get("max_flight_time", 0))
        features.append(aircraft.get("max_cycles", 0))
        features.append(aircraft.get("max_time_to_maintenance", 0))
        features.append(len(aircraft["planned_flights"]))
        features.append(self._calculate_max_ground_time(aircraft))
        features.append(1 if aircraft["initial_airport"] == dep_airport else 0)
        features.append(1 if aircraft["initial_airport"] == arr_airport else 0)
        features.append(DT_DF_time - STWS)  # 时间距窗口开始
        features.append(STWE - AT_DF_time)  # 时间距窗口结束
        features.append(len(all_flights))
        features.append(len(all_aircraft))
        features.append(np.random.rand())  # 模拟特征：飞机健康度
        features.append(np.random.rand())  # 模拟特征：任务负荷
        features.append(np.random.rand())  # 模拟特征：机型兼容度
        features.append(np.random.rand())  # 模拟特征：历史准点率
        # 保证维度到65
        while len(features) < 65:
            features.append(0.0)

        return np.array(features[:65])

    # ===============================================================
    # 二、飞机数量预测特征（27维）
    # ===============================================================
    def extract_count_features(self, disrupted_flight_id: str) -> np.ndarray:
        """
        提取预测飞机数量的特征（27维）
        """
        f = self.problem_data['flights'][disrupted_flight_id]
        all_aircraft = self.problem_data['aircraft']
        all_flights = self.problem_data['flights']

        STWS = min(ff["dep_time"] for ff in all_flights.values())
        STWE = max(ff["arr_time"] for ff in all_flights.values())
        TWL = STWE - STWS

        features = []

        # === 特征 1–7: 航班网络基本信息 ===
        CAN = len(all_aircraft) - 1
        ad_id = f.get("assigned_aircraft", None)
        NDF = len(all_aircraft[ad_id]["planned_flights"]) if ad_id in all_aircraft else 0
        NCF_SUM = sum(len(ac["planned_flights"]) for ac in all_aircraft.values())
        NCF_AVE = NCF_SUM / len(all_aircraft)
        features += [CAN, NDF, NCF_SUM, NCF_AVE, STWS, STWE, TWL]

        # === 特征 8: 干扰类型 one-hot ===
        DT_type = f.get("disruption_type", "turn_time")
        for dtype in ["turn_time", "airport_conflict", "airport_closure"]:
            features.append(1 if DT_type == dtype else 0)

        # === 特征 9–19: 与目标航班相关 ===
        DT_DF, AT_DF = f["dep_time"], f["arr_time"]
        DFT = AT_DF - DT_DF
        DT_DF_vs_STWS = DT_DF - STWS
        AT_DF_vs_STWE = AT_DF - STWE
        features += [DT_DF, AT_DF, DFT, DT_DF_vs_STWS, AT_DF_vs_STWE]

        # 模拟 vf 航班
        DT_VF, AT_VF, VFT = DT_DF + 0.5, AT_DF + 0.5, (AT_DF + 0.5) - (DT_DF + 0.5)
        DT_VF_vs_STWS, AT_VF_vs_STWE = DT_VF - STWS, AT_VF - STWE
        DST = np.mean([self._calculate_total_ground_time(ac) for ac in all_aircraft.values()])
        features += [DT_VF, AT_VF, VFT, DT_VF_vs_STWS, AT_VF_vs_STWE, DST]

        # === 特征 20–27: 航班密度和机场流量 ===
        ground_times = [self._calculate_total_ground_time(ac) for ac in all_aircraft.values()]
        flight_times = [ff["arr_time"] - ff["dep_time"] for ff in all_flights.values()]
        features += [min(ground_times), max(ground_times), np.mean(ground_times),
                     min(flight_times), max(flight_times), np.mean(flight_times)]

        dep_airport, arr_airport = f["dep_airport"], f["arr_airport"]
        CNA_1HB = sum(1 for ff in all_flights.values()
                      if ff["arr_airport"] == dep_airport and 0 <= (DT_DF - ff["arr_time"]) <= 1)
        CNA_1HF = sum(1 for ff in all_flights.values()
                      if ff["dep_airport"] == arr_airport and 0 <= (ff["dep_time"] - AT_DF) <= 1)
        features += [CNA_1HB, CNA_1HF]

        while len(features) < 27:
            features.append(0.0)

        return np.array(features[:27])

    # ===============================================================
    # 三、辅助计算函数
    # ===============================================================
    def _calculate_max_ground_time(self, aircraft: Dict) -> float:
        """计算最长地面停留时间"""
        flights = [self.problem_data['flights'][fid] for fid in aircraft['planned_flights']]
        flights.sort(key=lambda x: x['dep_time'])
        max_time = 0.0
        for i in range(len(flights) - 1):
            ground_time = flights[i + 1]['dep_time'] - flights[i]['arr_time']
            max_time = max(max_time, ground_time)
        return max_time

    def _calculate_total_ground_time(self, aircraft: Dict) -> float:
        """计算总地面停留时间"""
        flights = [self.problem_data['flights'][fid] for fid in aircraft['planned_flights']]
        flights.sort(key=lambda x: x['dep_time'])
        total = 0.0
        for i in range(len(flights) - 1):
            total += max(0.0, flights[i + 1]['dep_time'] - flights[i]['arr_time'])
        return total
