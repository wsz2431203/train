"""
集成列生成求解和排序模型训练 - 修正XML解析
根据实际文件格式调整ScenarioLoader
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRanker
from typing import List, Dict, Tuple, Set
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from pathlib import Path

# 假设已导入列生成求解器
from column_generation import ColumnGenerationSolver, VirtualFlight
from data_structures import Flight, Aircraft


class ScenarioLoader:
    """
    场景数据加载器
    从 input 文件夹读取 Schedule.xml, Parameters.xml, Maintenance.xml
    """
    
    # XML命名空间
    NAMESPACES = {
        'flt': 'http://generated.recoverymanager.sabre.com/flightSchedule',
        'inc': 'http://generated.recoverymanager.sabre.com/CommonDef',
        'mtc': 'http://generated.recoverymanager.sabre.com/mtcSchedule',
        'par': 'http://generated.recoverymanager.sabre.com/Parameters'
    }
    
    def __init__(self, input_folder: str):
        """
        Args:
            input_folder: input 文件夹路径，包含 Schedule.xml, Parameters.xml 等
        """
        self.input_folder = Path(input_folder)
        self.schedule_path = self.input_folder / "Schedule.xml"
        self.parameters_path = self.input_folder / "Parameters.xml"
        self.maintenance_path = self.input_folder / "Maintenance.xml"
        
        self._validate_paths()
    
    def _validate_paths(self):
        """验证必要文件是否存在"""
        if not self.input_folder.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {self.input_folder}")
        
        if not self.schedule_path.exists():
            raise FileNotFoundError(f"Schedule.xml 不存在: {self.schedule_path}")
        
        # Parameters.xml 和 Maintenance.xml 可选
    
    def load_schedule(self) -> pd.DataFrame:
        """
        从 Schedule.xml 加载航班时刻表
        XML结构: flt:fltList > flt:fltInfo
        
        Returns:
            包含航班信息的 DataFrame
        """
        tree = ET.parse(self.schedule_path)
        root = tree.getroot()
        
        flights_data = []
        
        # 使用命名空间查找
        for flight in root.findall('flt:fltInfo', self.NAMESPACES):
            try:
                # 提取基本信息
                leg_id = flight.find('flt:legID', self.NAMESPACES)
                tail = flight.find('flt:tail', self.NAMESPACES)
                fleet = flight.find('flt:fleet', self.NAMESPACES)
                
                dep_port = flight.find('flt:depPort', self.NAMESPACES)
                arr_port = flight.find('flt:arrPort', self.NAMESPACES)
                
                dep_time = flight.find('flt:depTime', self.NAMESPACES)
                arr_time = flight.find('flt:arrTime', self.NAMESPACES)
                
                block_time = flight.find('flt:blockTime', self.NAMESPACES)
                turn_time = flight.find('flt:turnTime', self.NAMESPACES)
                
                # 构建航班字典
                flight_info = {
                    'legID': int(leg_id.text) if leg_id is not None else None,
                    'tail': tail.text if tail is not None else None,
                    'fleet': fleet.text if fleet is not None else None,
                    'depPort': dep_port.text if dep_port is not None else None,
                    'arrPort': arr_port.text if arr_port is not None else None,
                    'depTime': pd.to_datetime(dep_time.text) if dep_time is not None else None,
                    'arrTime': pd.to_datetime(arr_time.text) if arr_time is not None else None,
                    'blockTime': float(block_time.text) if block_time is not None else 0,
                    'turnTime': float(turn_time.text) if turn_time is not None else 0,
                }
                
                # 可选字段
                seq_num = flight.find('flt:seqNum', self.NAMESPACES)
                flt_num = flight.find('flt:fltNum', self.NAMESPACES)
                dep_status = flight.find('flt:depStatus', self.NAMESPACES)
                arr_status = flight.find('flt:arrStatus', self.NAMESPACES)
                
                flight_info.update({
                    'seqNum': int(seq_num.text) if seq_num is not None else None,
                    'fltNum': flt_num.text if flt_num is not None else None,
                    'depStatus': dep_status.text if dep_status is not None else 'SCH',
                    'arrStatus': arr_status.text if arr_status is not None else 'SCH',
                })
                
                flights_data.append(flight_info)
                
            except Exception as e:
                print(f"    ⚠️ 解析航班失败: {e}")
                continue
        
        df = pd.DataFrame(flights_data)
        
        # 过滤掉已完成的航班（可选）
        if 'depStatus' in df.columns:
            # 保留未完成的航班
            df = df[df['depStatus'] != 'COMPLETED'].copy()
        
        return df
    
    def load_parameters(self) -> Dict:
    
        if not self.parameters_path.exists():
            print(f"  ⚠️ Parameters.xml 不存在，使用默认参数")
            return self._get_default_parameters()
        
        try:
            tree = ET.parse(self.parameters_path)
            root = tree.getroot()
            
            params = {}
            
            # 解析关键参数
            key_params = {
                # 成本相关参数
                'invertedFltSwapPenalty': 'swap_penalty',
                'invertedFltDelayPenalty': 'delay_penalty', 
                'invertedFltUnassignPenalty': 'cancel_penalty',
                'invertedFerryPenalty': 'ferry_penalty',
                'invertedDiversionPenalty': 'diversion_penalty',
                'invertedMtcUnassignPenalty': 'maintenance_cancel_penalty',
                'invertedMtcRetimePenalty': 'maintenance_retime_penalty',
                'invertedMtcRelocatePenalty': 'maintenance_relocate_penalty',
                'invertedEqpChangePenalty': 'equipment_change_penalty',
                
                # 时间窗口和阈值
                'currentTime': 'current_time',
                'reschedTimeStart': 'reschedule_start',
                'reschedTimeEnd': 'reschedule_end',
                'maxIncrementalDelay': 'max_delay',
                'dataDownloadStartThreshold': 'data_start_threshold',
                'dataDownloadEndThreshold': 'data_end_threshold',
                
                # 可行性配置
                'allowFltCancel': 'allow_cancellation',
                'allowFltDelay': 'allow_delay', 
                'allowMtcCancel': 'allow_maintenance_cancel',
                'allowFerries': 'allow_ferries',
                'allowDiversions': 'allow_diversions',
                'considerSpareAircraft': 'consider_spare_aircraft',
                'swapWithinOption': 'swap_within_option',
                
                # 其他重要参数
                'minimumPriorityLevel': 'min_priority_level',
                'imposeAirportCurfews': 'impose_curfews',
                'allowFltOverbooking': 'allow_overbooking',
                'allowOverFlying': 'allow_overflying'
            }
            
            # 提取参数值
            for xml_tag, param_key in key_params.items():
                element = root.find(f'par:{xml_tag}', self.NAMESPACES)
                if element is not None and element.text is not None:
                    text_value = element.text.strip()
                    
                    # 根据数据类型转换
                    if text_value.lower() in ['true', 'false']:
                        params[param_key] = text_value.lower() == 'true'
                    elif text_value.replace('.', '').replace('-', '').isdigit():
                        if '.' in text_value:
                            params[param_key] = float(text_value)
                        else:
                            params[param_key] = int(text_value)
                    elif 'T' in text_value:  # 日期时间格式
                        try:
                            params[param_key] = pd.to_datetime(text_value)
                        except:
                            params[param_key] = text_value
                    else:
                        params[param_key] = text_value
            
            # 设置默认值（如果某些关键参数缺失）
            params = self._set_defaults_if_missing(params)
            
            print(f"    ✓ 解析关键参数: 交换成本={params.get('swap_penalty')}, "
                f"延误成本={params.get('delay_penalty')}, "
                f"取消成本={params.get('cancel_penalty')}")
            
            return params
            
        except Exception as e:
            print(f"  ⚠️ 解析Parameters.xml失败: {e}")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> Dict:
        """获取默认参数值"""
        return {
            'swap_penalty': 800,
            'delay_penalty': 750,
            'cancel_penalty': 1000,
            'ferry_penalty': 500,
            'diversion_penalty': 927,
            'equipment_change_penalty': 50,
            'max_delay': 180,
            'allow_cancellation': True,
            'allow_delay': True,
            'allow_ferries': False,
            'allow_diversions': True,
            'consider_spare_aircraft': True,
            'swap_within_option': 2,
            'min_priority_level': 1,
            'impose_curfews': True
        }


    
    def load_maintenance(self) -> pd.DataFrame:
        """
        从 Maintenance.xml 加载维修信息
        XML结构: mtc:mtcListDynamic > mtc:mtcEvent
        
        Returns:
            包含维修信息的 DataFrame，如果文件不存在返回空 DataFrame
        """
        if not self.maintenance_path.exists():
            return pd.DataFrame()
        
        try:
            tree = ET.parse(self.maintenance_path)
            root = tree.getroot()
            
            maintenance_data = []
            
            # 使用命名空间查找
            for mtc_event in root.findall('mtc:mtcEvent', self.NAMESPACES):
                try:
                    tail = mtc_event.find('mtc:tail', self.NAMESPACES)
                    port = mtc_event.find('mtc:port', self.NAMESPACES)
                    event_status = mtc_event.find('mtc:eventStatus', self.NAMESPACES)
                    st_time = mtc_event.find('mtc:stTime', self.NAMESPACES)
                    en_time = mtc_event.find('mtc:enTime', self.NAMESPACES)
                    event_id = mtc_event.find('mtc:eventId', self.NAMESPACES)
                    activity_type = mtc_event.find('mtc:activityType', self.NAMESPACES)
                    
                    maint_info = {
                        'aircraft_id': tail.text if tail is not None else None,
                        'port': port.text if port is not None else None,
                        'start_time': pd.to_datetime(st_time.text) if st_time is not None else None,
                        'end_time': pd.to_datetime(en_time.text) if en_time is not None else None,
                        'status': event_status.text if event_status is not None else 'Planned',
                        'event_id': int(event_id.text) if event_id is not None else None,
                        'type': activity_type.text if activity_type is not None else 'O',
                    }
                    
                    maintenance_data.append(maint_info)
                    
                except Exception as e:
                    print(f"    ⚠️ 解析维修事件失败: {e}")
                    continue
            
            return pd.DataFrame(maintenance_data)
            
        except Exception as e:
            print(f"  ⚠️ 解析Maintenance.xml失败: {e}")
            return pd.DataFrame()
    
    def load_all(self) -> Dict:
        """
        加载所有数据
        
        Returns:
            包含所有数据的字典
        """
        print(f"  正在加载场景数据...")
        
        schedule = self.load_schedule()
        print(f"    ✓ 航班: {len(schedule)} 条")
        
        parameters = self.load_parameters()
        print(f"    ✓ 参数: {len(parameters)} 个")
        
        maintenance = self.load_maintenance()
        print(f"    ✓ 维修: {len(maintenance)} 条")
        
        return {
            'schedule': schedule,
            'parameters': parameters,
            'maintenance': maintenance,
            'input_folder': str(self.input_folder)
        }


class RankingLabelStrategy:
    """
    排序标注策略(论文3.3.3节)
    根据最优解中的飞机交换关系进行排序标注
    """
    
    def __init__(self):
        pass
    
    def label_aircraft_ranking(self, 
                               optimal_solution: Dict,
                               disrupted_flight_id: int,
                               original_aircraft_id: str,
                               all_aircraft: Set[str]) -> Dict[str, int]:
        """
        对候选飞机进行排序标注
        
        标注策略:
        1. 直接交换执飞目标不正常航班的飞机 -> 标注为1 (最高优先级)
        2. 与原飞机a_dis有交换关系的飞机 -> 标注为2
        3. 与集合A1和A2有交换关系的飞机 -> 标注为3
        4. 继续向外扩展...
        
        Args:
            optimal_solution: 列生成最优解
            disrupted_flight_id: 目标不正常航班ID
            original_aircraft_id: 原计划执飞的飞机ID (tail号)
            all_aircraft: 所有候选飞机ID集合
            
        Returns:
            {aircraft_id: ranking_label} 字典
        """
        # 从最优解中提取路径和飞机交换关系
        paths = optimal_solution.get('paths', [])
        
        # 构建交换关系图
        swap_graph = self._build_swap_graph(paths, disrupted_flight_id, 
                                            original_aircraft_id)
        
        # 初始化标注
        ranking_labels = {}
        labeled_aircraft = set()
        
        # ===== 第1层: 直接执飞目标不正常航班的飞机 =====
        A1 = set()
        for path in paths:
            if disrupted_flight_id in path.flights:
                if path.aircraft_id != original_aircraft_id:
                    A1.add(path.aircraft_id)
                    ranking_labels[path.aircraft_id] = 1
                    labeled_aircraft.add(path.aircraft_id)
        
        print(f"    A1 (标注1): {A1}")
        
        # ===== 第2层: 与a_dis有交换的飞机 =====
        A2 = set()
        if original_aircraft_id in swap_graph:
            for swap_aircraft in swap_graph[original_aircraft_id]:
                if swap_aircraft not in labeled_aircraft:
                    A2.add(swap_aircraft)
                    ranking_labels[swap_aircraft] = 2
                    labeled_aircraft.add(swap_aircraft)
        
        # 同时考虑A1中飞机与a_dis的交换
        for a1_aircraft in A1:
            if original_aircraft_id in swap_graph.get(a1_aircraft, set()):
                if original_aircraft_id not in labeled_aircraft:
                    A2.add(original_aircraft_id)
                    ranking_labels[original_aircraft_id] = 2
                    labeled_aircraft.add(original_aircraft_id)
        
        print(f"    A2 (标注2): {A2}")
        
        # ===== 第3+层: 逐层向外扩展 =====
        current_layer = A1.union(A2)
        layer_num = 3
        
        while current_layer and layer_num <= 10:  # 最多扩展到第10层
            next_layer = set()
            
            for aircraft in current_layer:
                if aircraft in swap_graph:
                    for swap_aircraft in swap_graph[aircraft]:
                        if swap_aircraft not in labeled_aircraft:
                            next_layer.add(swap_aircraft)
                            ranking_labels[swap_aircraft] = layer_num
                            labeled_aircraft.add(swap_aircraft)
            
            if next_layer:
                print(f"    A{layer_num} (标注{layer_num}): {next_layer}")
            
            current_layer = next_layer
            layer_num += 1
        
        # ===== 未被标注的飞机给予最低优先级 =====
        max_label = max(ranking_labels.values()) if ranking_labels else 0
        for aircraft_id in all_aircraft:
            if aircraft_id not in labeled_aircraft:
                ranking_labels[aircraft_id] = max_label + 1
        
        return ranking_labels
    
    def _build_swap_graph(self, paths: List, disrupted_flight_id: int,
                         original_aircraft_id: str) -> Dict[str, Set[str]]:
        """
        构建飞机交换关系图
        
        Returns:
            {aircraft_id: {交换对象aircraft_id集合}}
        """
        swap_graph = defaultdict(set)
        
        # 记录每个航班的原计划执飞飞机和实际执飞飞机
        flight_to_original = {}
        flight_to_actual = {}
        
        # 先找原计划
        for path in paths:
            if hasattr(path, 'recovery_type') and path.recovery_type == "original":
                for flight_id in path.flights:
                    flight_to_original[flight_id] = path.aircraft_id
        
        # 再找实际执飞
        for path in paths:
            for flight_id in path.flights:
                flight_to_actual[flight_id] = path.aircraft_id
        
        # 构建交换关系
        for flight_id, actual_aircraft in flight_to_actual.items():
            original_aircraft = flight_to_original.get(flight_id)
            
            if original_aircraft and original_aircraft != actual_aircraft:
                # 发生了交换
                swap_graph[original_aircraft].add(actual_aircraft)
                swap_graph[actual_aircraft].add(original_aircraft)
        
        return swap_graph
    
    


def compute_features(df: pd.DataFrame, disrupted_flight_id: int, 
                    ad: str, candidate_ac_list: List[str]) -> pd.DataFrame:
    """
    根据表A.1计算65个特征
    
    Args:
        df: 航班数据DataFrame
        disrupted_flight_id: 不正常航班ID
        ad: 原计划执飞飞机tail号
        candidate_ac_list: 候选飞机tail号列表
    
    Returns:
        特征DataFrame
    """
    fd = df[df["legID"] == disrupted_flight_id].iloc[0]
    depT_d = fd["depTime"]
    arrT_d = fd["arrTime"]
    depPort_d = fd["depPort"]
    arrPort_d = fd["arrPort"]

    features = []

    for ac in candidate_ac_list:
        subset = df[df["tail"] == ac]
        if subset.empty:
            continue

        # ========== 1-39特征 ==========
        DD = (arrT_d - depT_d).total_seconds() / 60
        SD_DF = df[df["tail"] == ad]["blockTime"].sum()
        SD_CF = subset["blockTime"].sum()
        DTAT = df[df["tail"] == ad]["turnTime"].mean()
        CTAT = subset["turnTime"].mean()
        CT_DT = DTAT - CTAT
        DST = df[df["tail"] == ad]["turnTime"].sum()

        CGT_DOA = subset[(subset["arrPort"] == depPort_d)]["turnTime"].sum()
        CD_DAM = int(depPort_d in subset["arrPort"].values)
        CD_AAM = int(arrPort_d in subset["arrPort"].values)
        CV_DAM = CD_DAM
        CV_AAM = CD_AAM

        def within_hours(df_ac, airport, ref_time, hours, before=True):
            delta = timedelta(hours=hours)
            if before:
                return int(any((df_ac["arrPort"] == airport) & 
                             (df_ac["arrTime"].between(ref_time - delta, ref_time))))
            else:
                return int(any((df_ac["arrPort"] == airport) & 
                             (df_ac["arrTime"].between(ref_time, ref_time + delta))))

        DDA_1HB = within_hours(subset, arrPort_d, arrT_d, 1, True)
        DDA_2HB = within_hours(subset, arrPort_d, arrT_d, 2, True)
        DDA_3HB = within_hours(subset, arrPort_d, arrT_d, 3, True)
        DDA_1HF = within_hours(subset, arrPort_d, arrT_d, 1, False)
        DDA_2HF = within_hours(subset, arrPort_d, arrT_d, 2, False)
        DDA_3HF = within_hours(subset, arrPort_d, arrT_d, 3, False)

        def min_time_diff(df_ac, airport, ref_time, before=True):
            if before:
                diffs = (ref_time - df_ac.loc[df_ac["arrPort"] == airport, 
                        "arrTime"]).dt.total_seconds() / 60
                diffs = diffs[diffs >= 0]
            else:
                diffs = (df_ac.loc[df_ac["arrPort"] == airport, "arrTime"] - 
                        ref_time).dt.total_seconds() / 60
                diffs = diffs[diffs >= 0]
            return diffs.min() if not diffs.empty else np.nan

        CMTB_DDA = min_time_diff(subset, arrPort_d, arrT_d, True)
        CMTF_DDA = min_time_diff(subset, arrPort_d, arrT_d, False)
        CGT_DDA = subset[subset["arrPort"] == arrPort_d]["turnTime"].sum()

        DOA_1HB = within_hours(subset, depPort_d, depT_d, 1, True)
        DOA_2HB = within_hours(subset, depPort_d, depT_d, 2, True)
        DOA_3HB = within_hours(subset, depPort_d, depT_d, 3, True)
        DOA_1HF = within_hours(subset, depPort_d, depT_d, 1, False)
        DOA_2HF = within_hours(subset, depPort_d, depT_d, 2, False)
        DOA_3HF = within_hours(subset, depPort_d, depT_d, 3, False)

        CMTB_DOA = min_time_diff(subset, depPort_d, depT_d, True)
        CMTF_DOA = min_time_diff(subset, depPort_d, depT_d, False)
        CDFS_AM = int(CD_DAM and CD_AAM)

        MAX_TI = (subset["arrTime"].max() - subset["depTime"].min()).total_seconds() / 60
        MIN_TI = (subset["arrTime"].min() - subset["depTime"].max()).total_seconds() / 60
        VFT = (arrT_d - depT_d).total_seconds() / 60 + fd["turnTime"]
        MAX_TI_VT = MAX_TI - VFT
        MIN_TI_VT = MIN_TI - VFT

        CA_VD_TI = min_time_diff(subset, depPort_d, depT_d, True)
        DA_CD_TI = min_time_diff(subset, arrPort_d, arrT_d, False)
        CST = subset["turnTime"].sum()
        CS_DS = int(any((subset["depTime"] <= depT_d) & (subset["arrTime"] >= depT_d)))

        # ========== 40-65: 航班网络基本信息 ==========
        CAN = len(candidate_ac_list) - 1
        DT = 0  # one-hot可在训练时处理
        
        ad_fleet = df[df["tail"] == ad]["fleet"].iloc[0] if not df[df["tail"] == ad].empty else None
        ac_fleet = subset["fleet"].iloc[0] if not subset.empty else None
        DC_TM = int(ac_fleet == ad_fleet) if ad_fleet and ac_fleet else 0
        DC_FM = DC_TM
        
        NDF = len(df[df["tail"] == ad])
        NCF = len(subset)
        DOC = subset["blockTime"].mean() if not subset.empty else 0
        COC = df[df["tail"] == ad]["blockTime"].mean() if not df[df["tail"] == ad].empty else 0
        CC_DC = COC - DOC
        DCP = 180
        CCP = 180
        CCP_DCP = CCP - DCP
        DOR = subset["blockTime"].sum() if not subset.empty else 0
        COR = df[df["tail"] == ad]["blockTime"].sum() if not df[df["tail"] == ad].empty else 0
        COR_DOR = COR - DOR
        DFTL = 1000
        CFTL = 1000
        CFTL_DFTL = CFTL - DFTL
        DFCL = 100
        CFCL = 100
        CFCL_DFCL = CFCL - DFCL
        DCH = 500
        CCH = 500
        CCH_DCH = CCH - DCH
        TDM = subset["turnTime"].sum()
        TCM = df[df["tail"] == ad]["turnTime"].sum()

        feature_dict = {
            "aircraft_id": ac,
            "DD": DD, "SD_DF": SD_DF, "SD_CF": SD_CF, "DTAT": DTAT, "CTAT": CTAT,
            "CT_DT": CT_DT, "DST": DST, "CGT_DOA": CGT_DOA, "CD_DAM": CD_DAM, 
            "CD_AAM": CD_AAM, "CV_DAM": CV_DAM, "CV_AAM": CV_AAM, 
            "DDA_1HB": DDA_1HB, "DDA_2HB": DDA_2HB, "DDA_3HB": DDA_3HB, 
            "DDA_1HF": DDA_1HF, "DDA_2HF": DDA_2HF, "DDA_3HF": DDA_3HF,
            "CMTB_DDA": CMTB_DDA, "CMTF_DDA": CMTF_DDA, "CGT_DDA": CGT_DDA, 
            "DOA_1HB": DOA_1HB, "DOA_2HB": DOA_2HB, "DOA_3HB": DOA_3HB, 
            "DOA_1HF": DOA_1HF, "DOA_2HF": DOA_2HF, "DOA_3HF": DOA_3HF, 
            "CMTB_DOA": CMTB_DOA, "CMTF_DOA": CMTF_DOA, "CDFS_AM": CDFS_AM,
            "MAX_TI": MAX_TI, "MIN_TI": MIN_TI, "VFT": VFT, 
            "MAX_TI_VT": MAX_TI_VT, "MIN_TI_VT": MIN_TI_VT, 
            "CA_VD_TI": CA_VD_TI, "DA_CD_TI": DA_CD_TI,
            "CST": CST, "CS_DS": CS_DS,
            "CAN": CAN, "DT": DT, "DC_TM": DC_TM, "DC_FM": DC_FM, 
            "NDF": NDF, "NCF": NCF, "DOC": DOC, "COC": COC, "CC_DC": CC_DC, 
            "DCP": DCP, "CCP": CCP, "CCP_DCP": CCP_DCP,
            "DOR": DOR, "COR": COR, "COR_DOR": COR_DOR, 
            "DFTL": DFTL, "CFTL": CFTL, "CFTL_DFTL": CFTL_DFTL, 
            "DFCL": DFCL, "CFCL": CFCL, "CFCL_DFCL": CFCL_DFCL,
            "DCH": DCH, "CCH": CCH, "CCH_DCH": CCH_DCH, "TDM": TDM, "TCM": TCM
        }
        features.append(feature_dict)

    return pd.DataFrame(features)


class IntegratedTrainer:
    """
    集成训练器:列生成求解 + 特征提取 + 排序模型训练
    """
    
    def __init__(self):
        self.labeling_strategy = RankingLabelStrategy()
        self.feature_names = [  # 65个特征名称
            "DD", "SD_DF", "SD_CF", "DTAT", "CTAT", "CT_DT", "DST",
            "CGT_DOA", "CD_DAM", "CD_AAM", "CV_DAM", "CV_AAM",
            "DDA_1HB", "DDA_2HB", "DDA_3HB", "DDA_1HF", "DDA_2HF", "DDA_3HF",
            "CMTB_DDA", "CMTF_DDA", "CGT_DDA", "DOA_1HB", "DOA_2HB", "DOA_3HB",
            "DOA_1HF", "DOA_2HF", "DOA_3HF", "CMTB_DOA", "CMTF_DOA", "CDFS_AM",
            "MAX_TI", "MIN_TI", "VFT", "MAX_TI_VT", "MIN_TI_VT",
            "CA_VD_TI", "DA_CD_TI", "CST", "CS_DS",
            "CAN", "DT", "DC_TM", "DC_FM", "NDF", "NCF", "DOC", "COC", "CC_DC",
            "DCP", "CCP", "CCP_DCP", "DOR", "COR", "COR_DOR",
            "DFTL", "CFTL", "CFTL_DFTL", "DFCL", "CFCL", "CFCL_DFCL",
            "DCH", "CCH", "CCH_DCH", "TDM", "TCM"
        ]
    def _display_top10_aircraft(self, ranking_labels: Dict[str, int], 
                            original_aircraft_id: str):
    
        # 打印标题和分隔线
        print(f"\n  {'='*56}")
        print(f"  排序标注结果 - Top 10 飞机")
        print(f"  {'='*56}")
        
        # 打印基本信息
        print(f"  原计划执飞飞机: {original_aircraft_id}")
        print(f"  总候选飞机数: {len(ranking_labels)}")
        print(f"  {'-'*56}")
        
        # 按标注值排序（值越小优先级越高）
        # 如果标注值相同，按飞机ID字母顺序排序
        sorted_aircraft = sorted(ranking_labels.items(), key=lambda x: (x[1], x[0]))
        
        # 打印表头
        print(f"  {'排名':<6} {'飞机ID':<15} {'标注值':<10} {'说明'}")
        print(f"  {'-'*56}")
        
        # 遍历前10个飞机
        for rank, (aircraft_id, label) in enumerate(sorted_aircraft[:10], 1):
            # 判断是否为原飞机，如果是则添加标记
            mark = "  (原飞机)" if aircraft_id == original_aircraft_id else ""
            
            # 根据标注值生成描述性说明
            if label == 1:
                desc = "直接执飞目标航班"
            elif label == 2:
                desc = "与原飞机有交换"
            elif label <= 5:
                desc = f"第{label}层交换关系"
            else:
                desc = "较远交换关系"
            
            # 打印每一行数据
            print(f"  {rank:<6} {aircraft_id:<15} {label:<10} {desc}{mark}")
        
        # 打印底部分隔线
        print(f"  {'='*56}\n")

    def process_single_scenario(self, 
                                input_folder: str,
                                solver: ColumnGenerationSolver = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        处理单个场景:加载数据+求解+特征提取+标注
        
        Args:
            input_folder: 包含 Schedule.xml, Parameters.xml 等的 input 文件夹路径
            solver: 可选的列生成求解器实例，如果为None则创建新实例
            
        Returns:
            (特征矩阵, 标签数组, 查询组大小列表)
        """
        print(f"\n{'='*60}")
        print(f"处理场景: {input_folder}")
        print(f"{'='*60}")
        
        # 步骤0: 加载场景数据
        print("\n[步骤0] 加载场景数据...")
        loader = ScenarioLoader(input_folder)
        scenario_data = loader.load_all()
        
        schedule_df = scenario_data['schedule']
        parameters = scenario_data['parameters']
        maintenance_df = scenario_data['maintenance']
        
        # 步骤1: 初始化或配置求解器
        if solver is None:
            solver = ColumnGenerationSolver()
        
        # TODO: 将加载的数据传入求解器
        solver.load_from_dataframe(schedule_df, maintenance_df, parameters)
        
        # 步骤2: 使用列生成求解获取最优解
        print("\n[步骤1] 列生成求解...")
        
        # 假设第一个不正常航班是目标
        target_disrupted_ids = list(solver.target_disrupted_flights)[:1]
        non_target_disrupted_ids = list(solver.non_target_disrupted_flights)
        
        optimal_solution = solver.solve(
            target_disrupted_ids=target_disrupted_ids,
            non_target_disrupted_ids=non_target_disrupted_ids,
            use_ml_selection=False,  # 训练阶段不使用ML
            max_iterations=50,
            scenario_path=input_folder 
        )
        
        if not optimal_solution['converged']:
            print("  ⚠️ 未收敛,跳过此场景")
            return None, None, []
        
        print(f"  ✓ 求解完成")
        
        # 步骤3: 对每个目标不正常航班进行标注和特征提取
        all_features = []
        all_labels = []
        group_sizes = []
        
        for disrupted_flight_id in target_disrupted_ids:
            print(f"\n[步骤2] 处理不正常航班 {disrupted_flight_id}")
            
            # 获取原计划执飞飞机
            disrupted_flight = solver.flights[disrupted_flight_id]
            original_aircraft_id = disrupted_flight.tail
            
            # 候选飞机集合(排除原飞机)
            candidate_aircraft = set(solver.aircraft.keys()) - {original_aircraft_id}
            candidate_aircraft_list = list(candidate_aircraft)
            
            # 2.1 排序标注
            print(f"  [2.1] 排序标注...")
            ranking_labels = self.labeling_strategy.label_aircraft_ranking(
                optimal_solution,
                disrupted_flight_id,
                original_aircraft_id,
                candidate_aircraft
            )
            self._display_top10_aircraft(ranking_labels, original_aircraft_id)
            # 2.2 特征提取（使用加载的 DataFrame）
            print(f"  [2.2] 特征提取...")
            features_df = compute_features(
                schedule_df,
                disrupted_flight_id,
                original_aircraft_id,
                candidate_aircraft_list
            )
            
            if features_df.empty:
                print("    ⚠️ 特征提取失败,跳过此不正常航班")
                continue
            
            # 2.3 合并特征和标签
            X = features_df[self.feature_names].values
            
            # 标签转换: 原始标注越小越好 -> LightGBM越大越好
            max_label = max(ranking_labels.values())
            y = []
            for _, row in features_df.iterrows():
                aircraft_id = row['aircraft_id']
                original_label = ranking_labels[aircraft_id]
                # 反转标签: 1->max, 2->max-1, ...
                inverted_label = max_label - original_label + 1
                y.append(inverted_label)
            
            all_features.append(X)
            all_labels.append(y)
            group_sizes.append(len(X))
            
            print(f"    ✓ 提取 {len(X)} 个候选飞机的特征")
        
        if not all_features:
            return None, None, []
        
        # 合并所有查询
        X_scenario = np.vstack(all_features)
        y_scenario = np.concatenate(all_labels)
        
        return X_scenario, y_scenario, group_sizes
    
    def train_ranking_model(self,
                           scenario_folders: List[str],
                           output_model_path: str = "aircraft_ranking_model.pkl"):
        """
        训练排序模型
        
        Args:
            scenario_folders: 训练场景的 input 文件夹路径列表
            output_model_path: 模型保存路径
        """
        print("\n" + "="*60)
        print("开始训练排序模型")
        print("="*60)
        
        all_X = []
        all_y = []
        all_groups = []
        
        # 处理所有场景
        for i, input_folder in enumerate(scenario_folders, 1):
            print(f"\n处理场景 {i}/{len(scenario_folders)}")
            
            X, y, groups = self.process_single_scenario(input_folder)
            
            if X is not None:
                all_X.append(X)
                all_y.append(y)
                all_groups.extend(groups)
                print(f"  ✓ 场景{i}完成: {X.shape[0]}个样本, {len(groups)}个查询")
            else:
                print(f"  ✗ 场景{i}跳过")
        
        if not all_X:
            print("\n❌ 没有有效的训练数据")
            return None
        
        # 合并所有数据
        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        
        print(f"\n{'='*60}")
        print("训练数据统计")
        print(f"{'='*60}")
        print(f"  总样本数: {X_train.shape[0]}")
        print(f"  特征维度: {X_train.shape[1]}")
        print(f"  查询组数: {len(all_groups)}")
        print(f"  标签范围: [{y_train.min()}, {y_train.max()}]")
        
        # 数据预处理
        X_train = np.nan_to_num(X_train, nan=0, posinf=999999, neginf=-999999)
        
        # 特征标准化(论文3.3.2节提到)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # 训练LambdaMART模型(使用LightGBM,参考论文表3.2)
        print(f"\n{'='*60}")
        print("训练LambdaMART模型")
        print(f"{'='*60}")
        
        model = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.01,  # 对应eta
            n_estimators=300,    # 对应n_estimators
            max_depth=15,        # 对应max_depth
            subsample=0.5,       # 对应subsample
            colsample_bytree=1.0,
            reg_alpha=0,
            reg_lambda=0.1,      # 对应gamma
            random_state=42,
            verbose=10
        )
        
        model.fit(
            X_train, 
            y_train, 
            group=all_groups,
            eval_set=[(X_train, y_train)],
            eval_group=[all_groups],
            eval_metric='ndcg',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)]
        )
        
        # 保存模型和标准化器
        model_package = {
            'model': model,
            'scaler': scaler,
            'feature_names': self.feature_names
        }
        
        with open(output_model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"\n✅ 模型已保存至: {output_model_path}")
        
        # 显示特征重要性
        self._show_feature_importance(model)
        
        return model
    
    def _show_feature_importance(self, model: LGBMRanker, top_k: int = 10):
        """显示特征重要性"""
        feature_importance = model.feature_importances_
        
        # 排序
        indices = np.argsort(feature_importance)[::-1]
        
        print(f"\n{'='*60}")
        print(f"Top {top_k} 重要特征")
        print(f"{'='*60}")
        
        for i, idx in enumerate(indices[:top_k], 1):
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"特征{idx}"
            importance = feature_importance[idx]
            print(f"  {i:2d}. {feat_name:15s}: {importance:8.2f}")


def main():
    """
    主程序:演示完整流程
    """
    print("="*60)
    print("集成列生成与排序模型训练")
    print("="*60)
    
    # 示例：指定多个场景的 input 文件夹路径
    scenario_folders = [
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AM2/819203/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819208/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819218/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819221/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819229/input',
    ]
    
    # 过滤存在的文件夹
    existing_folders = [f for f in scenario_folders if os.path.exists(f)]
    
    if not existing_folders:
        print("❌ 未找到有效的场景文件夹！")
        print("请确保以下路径存在:")
        for folder in scenario_folders:
            print(f"  - {folder}")
        return
    
    print(f"\n找到 {len(existing_folders)} 个有效场景:")
    for folder in existing_folders:
        print(f"  ✓ {folder}")
    
    # 创建训练器
    trainer = IntegratedTrainer()
    
    # 训练模型
    model = trainer.train_ranking_model(
        existing_folders,
        output_model_path="aircraft_ranking_model.pkl"
    )
    
    if model:
        print("\n✅ 训练完成!")
        print("\n模型可用于:")
        print("  1. 在列生成过程中进行飞机选择")
        print("  2. 加速求解过程")
        print("  3. 提供决策支持")
    else:
        print("\n❌ 训练失败!")


def batch_process_scenarios(root_folder: str, 
                            pattern: str = "*/input",
                            output_model_path: str = "aircraft_ranking_model.pkl"):
    """
    批量处理场景文件夹
    
    Args:
        root_folder: 根目录，包含多个场景
        pattern: 子文件夹模式，默认查找所有 input 文件夹
        output_model_path: 模型保存路径
    
    示例:
        batch_process_scenarios('D:/software/Engines/Rmops/rmops_rt/scenarios/AD')
    """
    from glob import glob
    
    root_path = Path(root_folder)
    
    # 查找所有匹配的 input 文件夹
    input_folders = list(root_path.glob(pattern))
    
    if not input_folders:
        print(f"❌ 在 {root_folder} 下未找到匹配 {pattern} 的文件夹")
        return
    
    print(f"找到 {len(input_folders)} 个场景:")
    for folder in input_folders:
        print(f"  - {folder}")
    
    # 训练
    trainer = IntegratedTrainer()
    model = trainer.train_ranking_model(
        [str(f) for f in input_folders],
        output_model_path=output_model_path
    )
    
    return model


if __name__ == "__main__":
    # 方式1: 手动指定场景列表
    main()
    '''
    # 方式2: 批量处理指定目录下的所有场景
    batch_process_scenarios(
        root_folder='D:/software/Engines/Rmops/rmops_rt/scenarios/AD',
        pattern='*/input',  # 匹配所有子文件夹下的 input 文件夹
        output_model_path='aircraft_ranking_model.pkl'
    )
    '''
