"""
列生成求解器模块(改进版-支持XML数据加载和自动识别不正常航班)
实现包含虚拟航班的列生成算法求解飞机恢复问题
基于论文4.2节的改进模型
"""
from typing import Dict, Set, List, Tuple, Optional
from data_structures import Flight, Aircraft, Maintenance, Slot, Path
from network_builder import NetworkBuilder
from label_algorithm import LabelSettingAlgorithm
from ml_selector import MLAircraftSelector
import pandas as pd
import numpy as np
import copy
import os
from pathlib import Path as FilePath
from datetime import datetime, timedelta


class VirtualFlight:
    """虚拟航班类"""
    
    def __init__(self, vf_id: int, affected_flights: List[int]):
        """
        初始化虚拟航班
        
        Args:
            vf_id: 虚拟航班ID
            affected_flights: 组成虚拟航班的受影响航班ID列表
        """
        self.id = vf_id
        self.affected_flights = affected_flights
        self.dep_airport = None
        self.arr_airport = None
        self.dep_time = None
        self.arr_time = None
        
    def set_time_space_info(self, flights: Dict[int, Flight]):
        """根据组成航班设置时空信息"""
        if self.affected_flights:
            first_flight = flights[self.affected_flights[0]]
            last_flight = flights[self.affected_flights[-1]]
            self.dep_airport = first_flight.dep_airport
            self.arr_airport = last_flight.arr_airport
            self.dep_time = first_flight.dep_time
            self.arr_time = last_flight.arr_time


class ColumnGenerationSolver:
    """改进的列生成求解器(支持虚拟航班和XML数据加载)"""
    
    def __init__(self):
        self.flights: Dict[int, Flight] = {}
        self.aircraft: Dict[str, Aircraft] = {}  # 改为str键(tail号)
        self.maintenances: Dict[int, Maintenance] = {}
        self.slots: Dict[int, Slot] = {}
        
        # 存储原计划路径
        self.original_paths: Dict[str, List[int]] = {}  # 改为str键
        
        # 航班分类
        self.target_disrupted_flights: Set[int] = set()
        self.non_target_disrupted_flights: Set[int] = set()
        self.affected_flights: Set[int] = set()
        self.normal_flights: Set[int] = set()
        
        # 虚拟航班
        self.virtual_flights: Dict[int, VirtualFlight] = {}
        self.cf_to_vf: Dict[int, int] = {}
        
        # 对偶变量
        self.dual_variables = {
            'beta_f': {},
            'beta_cf': {},
            'gamma': {},
            'delta': {},
            'theta': {}
        }
        
        # ML选择器和网络构建器
        self.ml_selector = MLAircraftSelector()
        self.network_builder = None
        self.initial_paths: List[Path] = []
        
    def is_disrupted_flight(self, row: pd.Series) -> Tuple[bool, str]:
        """
        判断航班是否为不正常航班
        
        Args:
            row: 航班数据行(DataFrame的一行)
            
        Returns:
            (是否不正常, 不正常原因)
        """
        reasons = []
        
        # 1. 检查起飞状态
        dep_status = row.get('depStatus', 'SCH')
        if pd.notna(dep_status) and dep_status not in ['SCH', 'COMPLETED']:
            reasons.append(f"起飞状态异常({dep_status})")
        
        # 2. 检查到达状态
        arr_status = row.get('arrStatus', 'SCH')
        if pd.notna(arr_status) and arr_status not in ['SCH', 'COMPLETED']:
            reasons.append(f"到达状态异常({arr_status})")
        
        # 3. 检查whatIf标志(假设性/虚拟航班)
        what_if = row.get('whatIf', False)
        if pd.notna(what_if) and str(what_if).lower() == 'true':
            reasons.append("假设性航班(whatIf=true)")
        
        # 4. 检查删除标志
        delete_ind = row.get('deleteInd', False)
        if pd.notna(delete_ind) and str(delete_ind).lower() == 'true':
            reasons.append("已删除航班(deleteInd=true)")
        
        # 5. 检查取消状态(CNL)
        if (pd.notna(dep_status) and dep_status == 'CNL') or \
           (pd.notna(arr_status) and arr_status == 'CNL'):
            reasons.append("航班取消(CNL)")
        
        is_disrupted = len(reasons) > 0
        reason_str = '; '.join(reasons) if reasons else '正常'
        
        return is_disrupted, reason_str
    
    def classify_disrupted_flights(self, schedule_df: pd.DataFrame, 
                                   target_flight_ids: List[int] = None) -> Dict:
        """
        自动分类不正常航班
        
        Args:
            schedule_df: 航班时刻表DataFrame
            target_flight_ids: 指定的目标不正常航班ID(可选)
            
        Returns:
            分类结果字典
        """
        print("\n" + "="*60)
        print("识别不正常航班")
        print("="*60)
        
        target_disrupted = []
        non_target_disrupted = []
        disruption_details = []
        
        for _, row in schedule_df.iterrows():
            leg_id = int(row['legID'])
            is_disrupted, reason = self.is_disrupted_flight(row)
            
            if is_disrupted:
                # 构建详细信息
                detail = {
                    'legID': leg_id,
                    'tail': row['tail'],
                    'depPort': row['depPort'],
                    'arrPort': row['arrPort'],
                    'depTime': row['depTime'],
                    'depStatus': row.get('depStatus', 'N/A'),
                    'arrStatus': row.get('arrStatus', 'N/A'),
                    'whatIf': row.get('whatIf', False),
                    'deleteInd': row.get('deleteInd', False),
                    'reason': reason
                }
                disruption_details.append(detail)
                
                # 分类为目标或非目标不正常航班
                if target_flight_ids and leg_id in target_flight_ids:
                    target_disrupted.append(leg_id)
                else:
                    non_target_disrupted.append(leg_id)
        
        # 如果没有指定目标航班,将所有不正常航班视为目标
        if not target_flight_ids and (target_disrupted or non_target_disrupted):
            all_disrupted = target_disrupted + non_target_disrupted
            target_disrupted = all_disrupted
            non_target_disrupted = []
        
        # 打印识别结果
        print(f"\n识别到 {len(disruption_details)} 个不正常航班:")
        print(f"  目标不正常航班: {len(target_disrupted)}")
        print(f"  非目标不正常航班: {len(non_target_disrupted)}")
        
        if disruption_details:
            print("\n详细信息:")
            for detail in disruption_details:
                print(f"  航班 {detail['legID']} ({detail['tail']})")
                print(f"    路线: {detail['depPort']} → {detail['arrPort']}")
                print(f"    起飞: {detail['depTime']} [{detail['depStatus']}]")
                print(f"    状态: {detail['reason']}")
        
        result = {
            'target_disrupted': target_disrupted,
            'non_target_disrupted': non_target_disrupted,
            'details': disruption_details,
            'total_disrupted': len(disruption_details)
        }
        
        # 更新内部状态
        self.target_disrupted_flights = set(target_disrupted)
        self.non_target_disrupted_flights = set(non_target_disrupted)
        
        print("\n" + "="*60)
        
        return result
        
    def load_from_dataframe(self, 
                           schedule_df: pd.DataFrame,
                           maintenance_df: pd.DataFrame = None,
                           parameters: Dict = None,
                           auto_identify_disruptions: bool = True,
                           target_flight_ids: List[int] = None):
        """
        从DataFrame加载数据
        
        Args:
            schedule_df: 航班时刻表DataFrame(来自ScenarioLoader)
            maintenance_df: 维修计划DataFrame
            parameters: 参数字典
            auto_identify_disruptions: 是否自动识别不正常航班
            target_flight_ids: 指定的目标不正常航班ID(可选)
        """
        print("\n" + "="*60)
        print("从DataFrame加载数据到求解器")
        print("="*60)
        
        # 1. 自动识别不正常航班(如果启用)
        if auto_identify_disruptions:
            disruption_result = self.classify_disrupted_flights(
                schedule_df, 
                target_flight_ids
            )
        
        # 2. 加载航班
        print("\n[1] 加载航班...")
        for _, row in schedule_df.iterrows():
            flight = Flight(
                id=int(row['legID']),
                dep_airport=row['depPort'],
                arr_airport=row['arrPort'],
                dep_time=row['depTime'],
                arr_time=row['arrTime'],
                duration=row['blockTime'],
                original_aircraft_id=row['tail'],
                turn_time=row.get("turnTime", 45),
                cancel_cost=float(parameters.get("invertedFltUnassignPenalty", 1000))
            ) 
            
            # 额外信息
            if 'seqNum' in row and pd.notna(row['seqNum']):
                flight.seq_num = int(row['seqNum'])
            if 'fltNum' in row and pd.notna(row['fltNum']):
                flight.flight_number = row['fltNum']
            if 'fleet' in row and pd.notna(row['fleet']):
                flight.fleet = row['fleet']
            if 'turnTime' in row and pd.notna(row['turnTime']):
                flight.turn_time = row['turnTime']
            
            # 保存状态信息
            if 'depStatus' in row:
                flight.dep_status = row['depStatus']
            if 'arrStatus' in row:
                flight.arr_status = row['arrStatus']
            if 'whatIf' in row:
                flight.what_if = row['whatIf']
            if 'deleteInd' in row:
                flight.delete_ind = row['deleteInd']
            
            self.add_flight(flight)
        
        print(f"  ✓ 加载 {len(self.flights)} 个航班")
        
        # 3. 构建飞机和原计划路径
        print("\n[2] 构建飞机和原计划路径...")
        aircraft_flights = schedule_df.groupby('tail')['legID'].apply(list).to_dict()
        
        for tail, flight_ids in aircraft_flights.items():
            if pd.isna(tail):
                continue
            
            # 按时间排序
            sorted_flights = sorted(
                flight_ids,
                key=lambda fid: self.flights[fid].dep_time
            )
            
            # 创建飞机对象
            first_flight = self.flights[sorted_flights[0]]
            aircraft = Aircraft(
                id=tail,
                initial_airport=first_flight.dep_airport,
                max_flight_time=1000.0,  # 默认值
                max_cycles=100,
                max_time_to_maintenance=48.0
            )
            aircraft.planned_flights = sorted_flights
            
            # 从参数中获取机型信息
            fleet_row = schedule_df[schedule_df['tail'] == tail].iloc[0]
            if 'fleet' in fleet_row and pd.notna(fleet_row['fleet']):
                aircraft.fleet = fleet_row['fleet']
            
            self.add_aircraft(aircraft)
            self.original_paths[tail] = sorted_flights
        
        print(f"  ✓ 创建 {len(self.aircraft)} 架飞机")
        
        # 4. 加载维修计划
        if maintenance_df is not None and not maintenance_df.empty:
            print("\n[3] 加载维修计划...")
            for idx, row in maintenance_df.iterrows():
                maintenance = Maintenance(
                    id=int(row['event_id']) if pd.notna(row['event_id']) else idx,
                    aircraft_id=row['aircraft_id'],
                    airport=row['port'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    maintenance_type=row.get('type', 'O')
                )
                self.add_maintenance(maintenance)
            print(f"  ✓ 加载 {len(self.maintenances)} 个维修计划")
        
        # 5. 应用参数(如果有)
        if parameters:
            print("\n[4] 应用参数...")
            # 根据实际参数格式进行配置
            # 例如: 时段容量、成本系数等
            print(f"  ✓ 应用 {len(parameters)} 个参数")
        
        print("\n" + "="*60)
        print("数据加载完成")
        print("="*60)
        print(f"  航班: {len(self.flights)}")
        print(f"  飞机: {len(self.aircraft)}")
        print(f"  维修: {len(self.maintenances)}")
        if auto_identify_disruptions:
            print(f"  目标不正常航班: {len(self.target_disrupted_flights)}")
            print(f"  非目标不正常航班: {len(self.non_target_disrupted_flights)}")
        
    def add_flight(self, flight: Flight):
        """添加航班"""
        self.flights[flight.id] = flight
        
    def add_aircraft(self, aircraft: Aircraft):
        """添加飞机"""
        self.aircraft[aircraft.id] = aircraft
        if hasattr(aircraft, 'planned_flights'):
            self.original_paths[aircraft.id] = aircraft.planned_flights.copy()
        
    def add_maintenance(self, maintenance: Maintenance):
        """添加检修计划"""
        self.maintenances[maintenance.id] = maintenance
        
    def add_slot(self, slot: Slot):
        """添加起降时段"""
        self.slots[slot.id] = slot
        
    def set_disrupted_flights(self, target_disrupted_ids: List[int], 
                             non_target_disrupted_ids: List[int] = None):
        """设置不正常航班"""
        self.target_disrupted_flights = set(target_disrupted_ids)
        self.non_target_disrupted_flights = set(non_target_disrupted_ids or [])
        
        print(f"\n目标不正常航班: {target_disrupted_ids}")
        print(f"非目标不正常航班: {non_target_disrupted_ids or []}")
        
    def identify_affected_flights(self):
        """识别受影响的航班(CF集合)"""
        self.affected_flights = set()
        
        for aircraft_id, flight_seq in self.original_paths.items():
            for i, fid in enumerate(flight_seq):
                if fid in self.non_target_disrupted_flights:
                    affected_range = [fid]
                    
                    # 向后查找受影响的航班
                    for j in range(i + 1, len(flight_seq)):
                        next_fid = flight_seq[j]
                        prev_fid = flight_seq[j - 1]
                        
                        curr_flight = self.flights[next_fid]
                        prev_flight = self.flights[prev_fid]
                        
                        # 检查是否因为前序航班问题而受影响
                        time_delta = (curr_flight.dep_time - prev_flight.arr_time).total_seconds() / 60
                        if (prev_flight.arr_airport != curr_flight.dep_airport or
                            time_delta < 30):
                            affected_range.append(next_fid)
                        else:
                            break
                    
                    self.affected_flights.update(affected_range)
        
        # 正常航班集合
        all_flights = set(self.flights.keys())
        self.normal_flights = all_flights - self.affected_flights
        
        print(f"\n识别受影响航班:")
        print(f"  受影响航班数: {len(self.affected_flights)}")
        print(f"  正常航班数: {len(self.normal_flights)}")
        
    def create_virtual_flights(self):
        """创建虚拟航班(VF集合)"""
        self.virtual_flights = {}
        self.cf_to_vf = {}
        vf_id = 10000
        
        aircraft_affected = {}
        for aircraft_id, flight_seq in self.original_paths.items():
            affected_in_path = [f for f in flight_seq if f in self.affected_flights]
            if affected_in_path:
                aircraft_affected[aircraft_id] = affected_in_path
        
        # 为每组受影响航班创建虚拟航班
        for aircraft_id, affected_list in aircraft_affected.items():
            segments = []
            current_segment = [affected_list[0]]
            
            for i in range(1, len(affected_list)):
                prev_idx = self.original_paths[aircraft_id].index(affected_list[i-1])
                curr_idx = self.original_paths[aircraft_id].index(affected_list[i])
                
                if curr_idx == prev_idx + 1:
                    current_segment.append(affected_list[i])
                else:
                    segments.append(current_segment)
                    current_segment = [affected_list[i]]
            
            segments.append(current_segment)
            
            for segment in segments:
                vf = VirtualFlight(vf_id, segment)
                vf.set_time_space_info(self.flights)
                self.virtual_flights[vf_id] = vf
                
                for cf_id in segment:
                    self.cf_to_vf[cf_id] = vf_id
                
                print(f"  创建虚拟航班 VF{vf_id}: 包含航班{segment}")
                vf_id += 1
        
        print(f"\n共创建 {len(self.virtual_flights)} 个虚拟航班")
        
    def check_path_feasibility(self, aircraft_id: str, flight_sequence: List[int], 
                              is_virtual_path: bool = False) -> Tuple[bool, str]:
        """检查路径可行性"""
        if not flight_sequence:
            return True, ""
            
        # 检查时空连续性
        for i in range(len(flight_sequence) - 1):
            curr_id = flight_sequence[i]
            next_id = flight_sequence[i + 1]
            
            if curr_id in self.flights:
                curr_flight = self.flights[curr_id]
            elif curr_id in self.virtual_flights:
                curr_flight = self.virtual_flights[curr_id]
            else:
                return False, f"未知航班ID: {curr_id}"
            
            if next_id in self.flights:
                next_flight = self.flights[next_id]
            elif next_id in self.virtual_flights:
                next_flight = self.virtual_flights[next_id]
            else:
                return False, f"未知航班ID: {next_id}"
            
            if curr_flight.arr_airport != next_flight.dep_airport:
                return False, f"空间不连续"
            
            min_turn_time = 30
            time_delta = (next_flight.dep_time - curr_flight.arr_time).total_seconds() / 60
            if time_delta < min_turn_time:
                return False, f"时间不连续"
        
        return True, ""
    
    def generate_initial_feasible_solution(self) -> List[Path]:
        """生成初始可行解(考虑虚拟航班)"""
        print("\n" + "="*60)
        print("生成初始可行解(含虚拟航班)")
        print("="*60)
        
        initial_paths = []
        
        for aircraft_id, original_sequence in self.original_paths.items():
            print(f"\n处理飞机 {aircraft_id}:")
            print(f"  原计划: {original_sequence}")
            
            has_affected = any(f in self.affected_flights for f in original_sequence)
            
            if not has_affected:
                is_feasible, _ = self.check_path_feasibility(aircraft_id, original_sequence)
                if is_feasible:
                    path = Path(aircraft_id, original_sequence,maintenances=[],cost=1000.0, used_slots=[])
                    path.recovery_type = "original"
                    path.is_virtual_path = False
                    initial_paths.append(path)
                    print(f"  ✓ 策略1: 原路径可行")
            else:
                modified_sequence = []
                i = 0
                
                while i < len(original_sequence):
                    fid = original_sequence[i]
                    
                    if fid in self.affected_flights:
                        vf_id = self.cf_to_vf.get(fid)
                        if vf_id:
                            vf = self.virtual_flights[vf_id]
                            modified_sequence.append(vf_id)
                            i += len(vf.affected_flights)
                        else:
                            modified_sequence.append(fid)
                            i += 1
                    else:
                        modified_sequence.append(fid)
                        i += 1
                
                is_feasible, _ = self.check_path_feasibility(
                    aircraft_id, modified_sequence, is_virtual_path=True
                )
                
                if is_feasible:
                    path = Path(aircraft_id, modified_sequence)
                    path.recovery_type = "virtual_flight"
                    path.is_virtual_path = True
                    initial_paths.append(path)
                    print(f"  ✓ 策略2: 虚拟航班恢复")
                    print(f"    修改后: {modified_sequence}")
                else:
                    cancelled_sequence = [f for f in original_sequence 
                                        if f not in self.affected_flights]
                    path = Path(aircraft_id, cancelled_sequence)
                    path.recovery_type = "cancellation"
                    path.is_virtual_path = False
                    path.cancelled_flights = [f for f in original_sequence 
                                             if f in self.affected_flights]
                    initial_paths.append(path)
                    print(f"  ✓ 策略3: 取消航班")
                    print(f"    取消: {path.cancelled_flights}")
        
        print(f"\n生成 {len(initial_paths)} 条初始路径")
        
        stats = {
            'original': sum(1 for p in initial_paths if p.recovery_type == "original"),
            'virtual': sum(1 for p in initial_paths if p.recovery_type == "virtual_flight"),
            'cancel': sum(1 for p in initial_paths if p.recovery_type == "cancellation")
        }
        print(f"  原路径: {stats['original']}, 虚拟航班: {stats['virtual']}, 取消: {stats['cancel']}")
        
        return initial_paths
        
    def initialize_dual_variables(self, selected_aircraft: Set[str]):
        """初始化对偶变量"""
        self.dual_variables['beta_f'] = {fid: 100.0 for fid in self.normal_flights}
        self.dual_variables['beta_cf'] = {fid: 100.0 for fid in self.affected_flights}
        self.dual_variables['gamma'] = {aid: -50.0 for aid in selected_aircraft}
        self.dual_variables['delta'] = {mid: 20.0 for mid in self.maintenances.keys()}
        self.dual_variables['theta'] = {sid: 5.0 for sid in self.slots.keys()}
        
    def calculate_arc_cost(self, aircraft_id: str, flight_id: int, 
                          is_virtual: bool = False) -> float:
        """计算弧的成本"""
        if is_virtual:
            vf = self.virtual_flights[flight_id]
            first_cf = vf.affected_flights[0]
            beta_cf = self.dual_variables['beta_cf'].get(first_cf, 0.0)
            
            slot_cost = 0.0
            for slot_id, slot in self.slots.items():
                if self._flight_uses_slot(vf, slot):
                    slot_cost += self.dual_variables['theta'].get(slot_id, 0.0)
            
            return - beta_cf - slot_cost
        else:
            if flight_id in self.normal_flights:
                flight = self.flights[flight_id]
                swap_cost = 50.0 if (hasattr(flight, 'original_aircraft_id') and 
                                    flight.original_aircraft_id != aircraft_id) else 0.0
                delay_cost = getattr(flight, 'delay', 0) * 10.0
                beta_f = self.dual_variables['beta_f'].get(flight_id, 0.0)
                
                slot_cost = 0.0
                for slot_id, slot in self.slots.items():
                    if self._flight_uses_slot(flight, slot):
                        slot_cost += self.dual_variables['theta'].get(slot_id, 0.0)
                
                return swap_cost + delay_cost - beta_f - slot_cost
            
            elif flight_id in self.affected_flights:
                flight = self.flights[flight_id]
                swap_cost = 50.0 if (hasattr(flight, 'original_aircraft_id') and 
                                    flight.original_aircraft_id != aircraft_id) else 0.0
                delay_cost = getattr(flight, 'delay', 0) * 10.0
                beta_cf = self.dual_variables['beta_cf'].get(flight_id, 0.0)
                
                slot_cost = 0.0
                for slot_id, slot in self.slots.items():
                    if self._flight_uses_slot(flight, slot):
                        slot_cost += self.dual_variables['theta'].get(slot_id, 0.0)
                
                return swap_cost + delay_cost - beta_cf - slot_cost
            
        return 0.0
    
    def _flight_uses_slot(self, flight, slot: Slot) -> bool:
        """判断航班是否使用起降时段"""
        if slot.start_time <= flight.dep_time <= slot.end_time:
            return True
        if slot.start_time <= flight.arr_time <= slot.end_time:
            return True
        return False
    
    def calculate_reduced_cost(self, aircraft_id: str, path: Path) -> float:
        """计算检验数"""
        reduced_cost = 0.0
        
        for flight_id in path.flights:
            is_virtual = flight_id in self.virtual_flights
            arc_cost = self.calculate_arc_cost(aircraft_id, flight_id, is_virtual)
            reduced_cost += arc_cost
        
        gamma_a = self.dual_variables['gamma'].get(aircraft_id, 0.0)
        reduced_cost -= gamma_a
        
        if hasattr(path, 'maintenances'):
            for maintenance_id in path.maintenances:
                delta_m = self.dual_variables['delta'].get(maintenance_id, 0.0)
                reduced_cost -= delta_m
        
        return reduced_cost
    
    def update_dual_variables(self, learning_rate: float = 0.05):
        """更新对偶变量"""
        for key in self.dual_variables['beta_f']:
            self.dual_variables['beta_f'][key] *= (1 - learning_rate)
        
        for key in self.dual_variables['beta_cf']:
            self.dual_variables['beta_cf'][key] *= (1 - learning_rate)
        
        for key in self.dual_variables['theta']:
            self.dual_variables['theta'][key] *= (1 - learning_rate * 0.5)
        
    def solve(self, target_disrupted_ids: List[int] = None, 
              non_target_disrupted_ids: List[int] = None,
              use_ml_selection: bool = True,
              max_iterations: int = 100,
              convergence_threshold: float = 1e-6,
              scenario_path: str = None,
              auto_detect: bool = True) -> Dict:
        """
        使用列生成求解飞机恢复问题
        
        Args:
            target_disrupted_ids: 目标不正常航班(可选,如果auto_detect=True则自动识别)
            non_target_disrupted_ids: 非目标不正常航班(可选)
            use_ml_selection: 是否使用ML选择子网络
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            scenario_path: 场景路径
            auto_detect: 是否使用自动识别的不正常航班
        """
        print("="*60)
        print("列生成求解(改进模型-含虚拟航班)")
        print("="*60)
        
        # 使用自动识别的结果或手动指定
        if auto_detect:
            # 使用自动识别的不正常航班
            print("\n使用自动识别的不正常航班")
        else:
            # 使用手动指定的不正常航班
            self.set_disrupted_flights(target_disrupted_ids or [], non_target_disrupted_ids)
            print("\n使用手动指定的不正常航班")
        
        self.identify_affected_flights()
        self.create_virtual_flights()
        self.initial_paths = self.generate_initial_feasible_solution()
        
        if use_ml_selection and self.target_disrupted_flights:
            first_target = list(self.target_disrupted_flights)[0]
            selected_aircraft = self.select_subnetwork_with_ml(first_target)
            print(f"\nML选择 {len(selected_aircraft)} 架飞机")
        else:
            selected_aircraft = set(self.aircraft.keys())
            print(f"\n使用全部 {len(selected_aircraft)} 架飞机")
        
        self.network_builder = NetworkBuilder(self.flights, self.aircraft)
        self.initialize_dual_variables(selected_aircraft)
        
        all_paths = self.initial_paths.copy()
        
        iteration = 0
        min_reduced_cost = 0.0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"迭代 {iteration}")
            print(f"{'='*60}")
            
            new_paths_found = False
            iteration_min_rc = float('inf')
            
            param_file = os.path.join(scenario_path, "Parameters.xml") if scenario_path else None
            label_algorithm = LabelSettingAlgorithm(
                self.network_builder,
                self.aircraft,
                self.dual_variables,
                param_file
            )
            
            for aircraft_id in selected_aircraft:
                paths = label_algorithm.solve(aircraft_id)
                
                if paths:
                    for path in paths:
                        rc = self.calculate_reduced_cost(aircraft_id, path)
                        path.reduced_cost = rc
                        iteration_min_rc = min(iteration_min_rc, rc)
                        
                        if rc < -convergence_threshold:
                            new_paths_found = True
                            all_paths.append(path)
                    
                    if new_paths_found:
                        neg_paths = [p for p in paths if p.reduced_cost < 0]
                        print(f"  飞机{aircraft_id}: {len(neg_paths)}条负检验数路径")
            
            min_reduced_cost = iteration_min_rc
            print(f"  最小检验数: {min_reduced_cost:.6f}")
            
            if not new_paths_found:
                print("\n收敛!")
                break
            
            self.update_dual_variables()
        
        result = {
            'iterations': iteration,
            'converged': not new_paths_found if iteration < max_iterations else False,
            'total_paths': len(all_paths),
            'virtual_flights_count': len(self.virtual_flights),
            'affected_flights_count': len(self.affected_flights),
            'target_disrupted_count': len(self.target_disrupted_flights),
            'non_target_disrupted_count': len(self.non_target_disrupted_flights),
            'paths': all_paths,
            'virtual_flights': self.virtual_flights,
            'min_reduced_cost': min_reduced_cost,
            'disrupted_flights': {
                'target': list(self.target_disrupted_flights),
                'non_target': list(self.non_target_disrupted_flights)
            }
        }
        
        print(f"\n{'='*60}")
        print("求解完成")
        print(f"{'='*60}")
        print(f"  迭代次数: {iteration}")
        print(f"  收敛: {result['converged']}")
        print(f"  生成路径: {result['total_paths']}")
        print(f"  虚拟航班: {result['virtual_flights_count']}")
        print(f"  受影响航班: {result['affected_flights_count']}")
        print(f"  目标不正常航班: {result['target_disrupted_count']}")
        print(f"  非目标不正常航班: {result['non_target_disrupted_count']}")
        
        return result
    
    def select_subnetwork_with_ml(self, disrupted_flight_id: int) -> Set[str]:
        """使用ML选择子网络"""
        case = {
            'aircraft': {aid: {
                'id': aid,
                'initial_airport': ac.initial_airport,
                'max_flight_time': ac.max_flight_time,
                'max_cycles': ac.max_cycles,
                'max_time_to_maintenance': ac.max_time_to_maintenance,
                'planned_flights': ac.planned_flights
            } for aid, ac in self.aircraft.items()},
            'flights': {fid: {
                'id': fid,
                'dep_airport': f.dep_airport,
                'arr_airport': f.arr_airport,
                'dep_time': f.dep_time,
                'arr_time': f.arr_time,
                'duration': f.duration,
                'disruption_type': 'turn_time'
            } for fid, f in self.flights.items()},
            'time_window': 24.0,
            'disrupted_flight_id': disrupted_flight_id
        }
        
        aircraft_ranking = self.ml_selector.predict_aircraft_ranking(
            case,
            disrupted_flight_id
        )
        aircraft_count = self.ml_selector.predict_aircraft_count(
            case,
            disrupted_flight_id
        )
        
        selected_aircraft = set()
        for i in range(min(aircraft_count, len(aircraft_ranking))):
            selected_aircraft.add(aircraft_ranking[i][0])
            
        return selected_aircraft
    
    def export_disruption_report(self, output_path: str = None) -> pd.DataFrame:
        """
        导出不正常航班报告
        
        Args:
            output_path: 输出文件路径(可选)
            
        Returns:
            不正常航班报告DataFrame
        """
        report_data = []
        
        all_disrupted = self.target_disrupted_flights.union(self.non_target_disrupted_flights)
        
        for flight_id in all_disrupted:
            if flight_id not in self.flights:
                continue
                
            flight = self.flights[flight_id]
            
            report_data.append({
                '航班ID': flight_id,
                '机尾号': flight.original_aircraft_id if hasattr(flight, 'original_aircraft_id') else 'N/A',
                '航班号': flight.flight_number if hasattr(flight, 'flight_number') else 'N/A',
                '起飞机场': flight.dep_airport,
                '到达机场': flight.arr_airport,
                '计划起飞': flight.dep_time,
                '计划到达': flight.arr_time,
                '起飞状态': flight.dep_status if hasattr(flight, 'dep_status') else 'N/A',
                '到达状态': flight.arr_status if hasattr(flight, 'arr_status') else 'N/A',
                'whatIf': flight.what_if if hasattr(flight, 'what_if') else False,
                'deleteInd': flight.delete_ind if hasattr(flight, 'delete_ind') else False,
                '类型': '目标不正常' if flight_id in self.target_disrupted_flights else '非目标不正常',
                '是否受影响': '是' if flight_id in self.affected_flights else '否'
            })
        
        report_df = pd.DataFrame(report_data)
        
        if output_path:
            report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n不正常航班报告已导出到: {output_path}")
        
        return report_df
    
    def get_disruption_summary(self) -> Dict:
        """
        获取不正常航班汇总信息
        
        Returns:
            汇总信息字典
        """
        all_disrupted = self.target_disrupted_flights.union(self.non_target_disrupted_flights)
        
        # 按飞机统计
        aircraft_disruptions = {}
        for flight_id in all_disrupted:
            if flight_id in self.flights:
                flight = self.flights[flight_id]
                tail = flight.original_aircraft_id if hasattr(flight, 'original_aircraft_id') else 'Unknown'
                
                if tail not in aircraft_disruptions:
                    aircraft_disruptions[tail] = {
                        'target': 0,
                        'non_target': 0,
                        'affected': 0,
                        'flights': []
                    }
                
                if flight_id in self.target_disrupted_flights:
                    aircraft_disruptions[tail]['target'] += 1
                else:
                    aircraft_disruptions[tail]['non_target'] += 1
                
                if flight_id in self.affected_flights:
                    aircraft_disruptions[tail]['affected'] += 1
                
                aircraft_disruptions[tail]['flights'].append(flight_id)
        
        # 按状态统计
        status_counts = {}
        for flight_id in all_disrupted:
            if flight_id in self.flights:
                flight = self.flights[flight_id]
                
                dep_status = flight.dep_status if hasattr(flight, 'dep_status') else 'Unknown'
                arr_status = flight.arr_status if hasattr(flight, 'arr_status') else 'Unknown'
                
                key = f"{dep_status}/{arr_status}"
                status_counts[key] = status_counts.get(key, 0) + 1
        
        summary = {
            'total_disrupted': len(all_disrupted),
            'target_disrupted': len(self.target_disrupted_flights),
            'non_target_disrupted': len(self.non_target_disrupted_flights),
            'affected_flights': len(self.affected_flights),
            'normal_flights': len(self.normal_flights),
            'aircraft_disruptions': aircraft_disruptions,
            'status_counts': status_counts,
            'virtual_flights_created': len(self.virtual_flights)
        }
        
        return summary
    
    def print_disruption_summary(self):
        """打印不正常航班汇总信息"""
        summary = self.get_disruption_summary()
        
        print("\n" + "="*60)
        print("不正常航班汇总")
        print("="*60)
        
        print(f"\n总体统计:")
        print(f"  总航班数: {len(self.flights)}")
        print(f"  不正常航班: {summary['total_disrupted']}")
        print(f"    ├─ 目标不正常: {summary['target_disrupted']}")
        print(f"    └─ 非目标不正常: {summary['non_target_disrupted']}")
        print(f"  受影响航班: {summary['affected_flights']}")
        print(f"  正常航班: {summary['normal_flights']}")
        print(f"  虚拟航班: {summary['virtual_flights_created']}")
        
        print(f"\n按飞机统计:")
        for tail, data in summary['aircraft_disruptions'].items():
            print(f"  {tail}:")
            print(f"    目标不正常: {data['target']}, 非目标: {data['non_target']}, 受影响: {data['affected']}")
        
        print(f"\n按状态统计 (起飞/到达):")
        for status, count in summary['status_counts'].items():
            print(f"  {status}: {count}")
        
        print("="*60)
