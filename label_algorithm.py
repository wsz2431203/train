"""
标签设置算法模块(改进版 - 支持虚拟航班)
包含 slot、延误、交换等真实成本逻辑,以及虚拟航班处理
"""

from typing import List, Dict, Optional, Set
from collections import defaultdict
from data_structures import Label, Path, Aircraft
from network_builder import NetworkBuilder
import xml.etree.ElementTree as ET


class LabelSettingAlgorithm:
    """标签设置算法(带成本扩展和虚拟航班支持)"""
    
    def __init__(self, network_builder: NetworkBuilder, 
                 aircraft: Dict[int, Aircraft],
                 dual_variables: Dict,
                 param_file: str,
                 virtual_flights: Dict = None,
                 affected_flights_map: Dict = None):
        """
        初始化标签设置算法
        
        Args:
            network_builder: 网络构建器
            aircraft: 飞机字典
            dual_variables: 对偶变量字典(含 beta_f, beta_cf, gamma, delta, theta)
            param_file: XML 参数文件路径
            virtual_flights: 虚拟航班字典 {虚拟航班id: [受影响航班id列表]}
            affected_flights_map: 受影响航班到虚拟航班的映射 {受影响航班id: [虚拟航班id列表]}
        """
        self.network_builder = network_builder
        self.aircraft = aircraft
        self.dual_variables = dual_variables
        self.parameters = self._load_parameters(param_file)
        self.virtual_flights = virtual_flights or {}
        self.affected_flights_map = affected_flights_map or {}
        
    def _load_parameters(self, xml_path: str) -> Dict[str, float]:
        """从 XML 文件读取参数"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = {"par": "http://generated.recoverymanager.sabre.com/Parameters"}
        params = {}
        for child in root.findall("par:*", ns):
            params[child.tag.split("}")[1]] = float(child.text) if child.text and child.text.replace('.', '', 1).isdigit() else child.text
        return params

    def solve(self, aircraft_id: int) -> List[Path]:
        """为指定飞机求解定价子问题"""
        nodes, edges = self.network_builder.build_network(aircraft_id)
        label_sets = defaultdict(list)
        
        aircraft = self.aircraft[aircraft_id]
        start_label = Label(
            l1=-self.dual_variables['gamma'].get(aircraft_id, 0),
            l2=0.0,
            l3=aircraft.max_flight_time,
            l4=aircraft.max_cycles,
            l5=aircraft.max_time_to_maintenance,
            l6=None,
            l7=set(),  # 新增: 虚拟航班集合
            path=[],
            used_slots=set()
        )
        label_sets[-1] = [start_label]
        
        sorted_nodes = self.network_builder.topological_sort(nodes, edges)
        
        for node_id in sorted_nodes:
            if node_id == -2:
                continue
                
            for next_node_id in edges.get(node_id, []):
                # 检查虚拟航班连接限制
                if not self._can_connect(node_id, next_node_id, nodes):
                    continue
                    
                for label in label_sets[node_id]:
                    # 路径可行性检查
                    if not self._is_path_feasible(label, next_node_id, nodes):
                        continue
                    
                    new_label = self._create_extended_label(
                        node_id, next_node_id, label, nodes
                    )
                    if new_label and not self._is_dominated(new_label, label_sets[next_node_id]):
                        label_sets[next_node_id].append(new_label)
                        self._remove_dominated_labels(new_label, label_sets[next_node_id])
        
        generated_paths = []
        for label in label_sets.get(-2, []):
            if label.l1 < -1e-6:
                path = Path(
                    aircraft_id=aircraft_id,
                    flights=label.path,
                    maintenances=[label.l6] if label.l6 else [],
                    cost=label.l1,
                    used_slots=dict.fromkeys(label.used_slots, 1)
                )
                generated_paths.append(path)
        return generated_paths

    def _can_connect(self, from_id: int, to_id: int, nodes: Dict) -> bool:
        """
        判断两个节点是否可以连接(考虑虚拟航班限制)
        虚拟航班节点不可连接组成该虚拟航班的受影响航班节点
        """
        if from_id == -1 or to_id == -2:
            return True
        
        from_node = nodes.get(from_id)
        to_node = nodes.get(to_id)
        
        if not from_node or not to_node:
            return True
        
        # 检查: 虚拟航班不能连接其组成的受影响航班
        if from_node['type'] == 'virtual_flight' and to_node['type'] == 'flight':
            from_flight = from_node['flight']
            to_flight = to_node['flight']
            
            # 如果 from 是虚拟航班,且 to 是该虚拟航班的组成部分
            if hasattr(from_flight, 'is_virtual') and from_flight.is_virtual:
                affected_flights = self.virtual_flights.get(from_flight.id, [])
                if to_flight.id in affected_flights:
                    return False
        
        return True

    def _is_path_feasible(self, label: Label, next_node_id: int, nodes: Dict) -> bool:
        """
        路径可行性检查
        规则: 路径中不可同时包含受影响航班及其组成的虚拟航班
        """
        next_node = nodes.get(next_node_id)
        if not next_node or next_node['type'] == 'end':
            return True
        
        next_flight = next_node.get('flight')
        if not next_flight:
            return True
        
        # 规则①: 若节点j代表某受影响航班cf,当cf参与构成集合vfl7_i中的任意一个虚拟航班时,
        # 则节点j不能成为节点i的后继节点,该路径不可行
        if next_node['type'] == 'affected_flight' or (hasattr(next_flight, 'is_affected') and next_flight.is_affected):
            affected_flight_id = next_flight.id
            # 检查该受影响航班是否参与构成了标签中已有的虚拟航班
            virtual_flights_in_path = label.l7
            for vf_id in virtual_flights_in_path:
                affected_flights = self.virtual_flights.get(vf_id, [])
                if affected_flight_id in affected_flights:
                    return False
        
        return True

    def _create_extended_label(self, from_id: int, to_id: int, 
                              label: Label, nodes: Dict) -> Optional[Label]:
        """从当前标签扩展到下一个节点"""
        to_node = nodes[to_id]
        if to_node['type'] == 'end':
            return label
        if to_node['type'] not in ['flight', 'virtual_flight', 'affected_flight', 'normal_flight']:
            return None
        
        flight = to_node['flight']
        new_l3 = label.l3 - flight.duration
        new_l4 = label.l4 - 1
        new_l5 = label.l5 - flight.duration
        if new_l3 < 0 or new_l4 < 0 or new_l5 < 0:
            return None
        
        # === 更新 l7 (虚拟航班集合) ===
        new_l7 = label.l7.copy()
        
        # 规则②: 若节点j代表某虚拟航班vf,则vfl7_j = vfl7_i ∪ {vf}
        if to_node['type'] == 'virtual_flight' or (hasattr(flight, 'is_virtual') and flight.is_virtual):
            new_l7.add(flight.id)
        # 规则③: 若节点j代表其他航班f,则vfl7_j = vfl7_i (集合不做调整)
        
        # === 成本计算 ===
        # 根据航班类型选择正确的对偶变量
        if to_node['type'] == 'virtual_flight' or (hasattr(flight, 'is_virtual') and flight.is_virtual):
            # 虚拟航班使用 beta_cf (因为虚拟航班代表受影响航班组合)
            beta = self.dual_variables.get('beta_cf', {}).get(flight.id, 0)
        elif flight.id in self.dual_variables.get('beta_cf', {}):
            # 受影响航班使用 beta_cf
            beta = self.dual_variables['beta_cf'].get(flight.id, 0)
        else:
            # 正常航班使用 beta_f
            beta = self.dual_variables.get('beta_f', {}).get(flight.id, 0)
        
        theta = self.dual_variables.get('theta', {})

        # 1. 延误成本
        delay_td = flight.actual_departure - flight.scheduled_departure
        delay_minutes = max(0, delay_td.total_seconds() / 60.0)  # 转换为分钟
        c_delay = delay_minutes * self.parameters.get("invertedFltDelayPenalty", 750)

        # 2. 飞机交换成本
        c_swap = 0
        if label.path:
            prev_flight_id = label.path[-1]
            prev_node = nodes.get(prev_flight_id)
            if prev_node and prev_node.get('flight'):
                prev_flight = prev_node['flight']
                if prev_flight.aircraft_id != flight.aircraft_id:
                    c_swap = self.parameters.get("invertedFltSwapPenalty", 800)

        # 3. Slot 成本
        c_slot = 0
        new_used_slots = label.used_slots.copy()
        if hasattr(flight, 'slot_id') and flight.slot_id:
            slot_id = flight.slot_id
            new_used_slots.add(slot_id)
            c_slot = theta.get(slot_id, 0) + self.parameters.get("invertedSlotTimePenalty", 500)

        # === 更新总成本 ===
        new_l1 = label.l1 - beta + c_delay + c_swap + c_slot

        new_label = Label(
            l1=new_l1,
            l2=label.l2,
            l3=new_l3,
            l4=new_l4,
            l5=new_l5,
            l6=label.l6,
            l7=new_l7,  # 更新虚拟航班集合
            path=label.path + [flight.id],
            used_slots=new_used_slots
        )
        return new_label
    
    def _is_dominated(self, label: Label, label_list: List[Label]) -> bool:
        """检查标签是否被支配"""
        for existing_label in label_list:
            if existing_label.dominates(label):
                return True
        return False
    
    def _remove_dominated_labels(self, new_label: Label, label_list: List[Label]):
        """移除被新标签支配的标签"""
        i = len(label_list) - 1
        while i >= 0:
            if new_label.dominates(label_list[i]) and label_list[i] != new_label:
                label_list.pop(i)
            i -= 1
