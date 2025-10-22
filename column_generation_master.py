import heapq # 用于Dijkstra算法（最短路径问题）中的优先队列
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import gurobipy as gp # 导入Gurobi库，用于构建和求解线性规划问题
from gurobipy import GRB, Model, LinExpr # 确保导入了 LinExpr

import pandas as pd # 导入pandas，用于is_disrupted_flight中的row类型提示
from typing import Tuple # 用于is_disrupted_flight的返回类型提示
import random # 导入random库用于生成随机成本
import itertools # 用于生成唯一的label_id


# --- 用于解析XML数据和数据模型的辅助函数和类 ---

def parse_datetime(dt_str):
    """
    将ISO 8601格式的日期时间字符串解析为datetime对象。
    """
    return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%fZ")

def calculate_delay_cost(actual_dep_time, scheduled_benchmark_dep_time, inverted_flt_delay_penalty=0):
    """
    计算航班延误成本。
    延误成本 = 延误分钟数 * 每分钟延误惩罚。
    使用 scheduled_benchmark_dep_time 作为基准（例如 pub_sched_dep_time）。
    """
    delay_minutes = max(0, int((actual_dep_time - scheduled_benchmark_dep_time).total_seconds() / 60))
    return delay_minutes * inverted_flt_delay_penalty

def calculate_swap_cost(original_aircraft_id, current_aircraft_id, inverted_flt_swap_penalty=501):
    """
    计算飞机调换成本。
    如果航班的原始飞机与当前飞机不同，则产生调换惩罚。
    """
    return inverted_flt_swap_penalty if original_aircraft_id != current_aircraft_id else 0

class Flight:
    """
    表示一个航班。
    """
    def __init__(self, leg_id, seq_num, dep_port, arr_port, dep_time, arr_time,
                pub_sched_dep_time, pub_sched_arr_time, turn_time, block_time,
                stage, fleet, crew_rating, avg_fare, flt_dt, flt_num, eqp,
                suffix, airline, dep_status, arr_status, tail, service, what_if,
                delete_ind):
        self.leg_id = leg_id
        self.seq_num = int(seq_num)
        self.origin = dep_port
        self.destination = arr_port
        self.scheduled_dep_time = parse_datetime(dep_time) # 用于内部排班，例如计算 earliest_event_ready_time
        self.scheduled_arr_time = parse_datetime(arr_time)
        self.pub_sched_dep_time = parse_datetime(pub_sched_dep_time) # 用于计算延误成本的基准
        self.pub_sched_arr_time = parse_datetime(pub_sched_arr_time)
        self.turn_time = int(turn_time) # 停站时间，单位分钟
        self.block_time = int(block_time) # 总飞行时间，单位分钟
        self.flight_time = self.scheduled_arr_time - self.scheduled_dep_time # timedelta对象，表示飞行时长
        self.stage = int(stage)
        self.fleet = fleet
        self.crew_rating = int(crew_rating)
        self.avg_fare = float(avg_fare)
        self.flt_dt = parse_datetime(flt_dt)
        self.flt_num = flt_num
        self.eqp = eqp
        self.suffix = suffix
        self.airline = airline
        self.dep_status = dep_status
        self.arr_status = arr_status
        self.tail = tail # 这是航班最初分配的飞机ID
        self.service = service
        self.what_if = what_if == 'true'
        self.delete_ind = delete_ind == 'true'
        # self.processed_in_initial_solution = False # 新增标志，用于初始解生成
        self.is_disrupted, self.disruption_reason = self._check_disruption_status() # 预计算航班中断状态
        self.operating_cost = random.randint(100, 200) # 为每个航班随机生成一个运营成本

    def _check_disruption_status(self) -> Tuple[bool, str]:
        """
        判断航班是否为不正常航班 (内部方法，基于Flight实例的属性)
        
        Returns:
            (是否不正常, 不正常原因)
        """
        reasons = []
        
        # 1. 检查起飞状态
        if self.dep_status not in ['SCH', 'COMPLETED', 'ETD']: # 增加ETD为正常状态
            reasons.append(f"起飞状态异常({self.dep_status})")
        
        # 2. 检查到达状态
        if self.arr_status not in ['SCH', 'COMPLETED', 'ETA']: # 增加ETA为正常状态
            reasons.append(f"到达状态异常({self.arr_status})")
        
        # 3. 检查whatIf标志(假设性/虚拟航班)
        if self.what_if: # what_if已经是布尔值
            reasons.append("假设性航班(whatIf=true)")
        
        # 4. 检查删除标志
        if self.delete_ind: # delete_ind已经是布尔值
            reasons.append("已删除航班(deleteInd=true)")
        
        # 5. 检查取消状态(CNL)
        if self.dep_status == 'CNL' or self.arr_status == 'CNL':
            reasons.append("航班取消(CNL)")
        
        is_disrupted = len(reasons) > 0
        reason_str = '; '.join(reasons) if reasons else '正常'
        
        return is_disrupted, reason_str

    def __repr__(self):
        return f"Flight(ID={self.leg_id}, {self.origin}-{self.destination}, Tail={self.tail}, Disrupted={self.is_disrupted}, OpCost={self.operating_cost})"

class Aircraft:
    """
    表示一架飞机。
    """
    def __init__(self, aircraft_id, initial_location, initial_time,
                initial_available_flying_hours=999999, # 默认可用飞行小时数很大
                initial_remaining_cycles=9999,       # 默认剩余循环数很大
                initial_time_to_next_maintenance=999999): # 默认距离下次维护时间很大，单位分钟
        self.id = aircraft_id
        self.initial_location = initial_location
        self.initial_time = initial_time # datetime对象
        self.initial_available_flying_hours = initial_available_flying_hours
        self.initial_remaining_cycles = initial_remaining_cycles
        self.initial_time_to_next_maintenance = initial_time_to_next_maintenance # 单位分钟

    def __repr__(self):
        return f"Aircraft(ID={self.id}, Base={self.initial_location})"

class MaintenancePlan:
    """
    表示一个维护计划。
    """
    def __init__(self, event_id, tail, port, event_status, st_time, en_time,
                mtc_group_id, pre_time, post_time, is_spare, priority, what_if):
        self.id = event_id
        self.aircraft_id = tail # 适用于此维护计划的飞机ID
        self.required_airport = port
        self.event_status = event_status
        self.start_time = parse_datetime(st_time)
        self.end_time = parse_datetime(en_time)
        self.duration = (self.end_time - self.start_time) # timedelta对象，维护时长
        self.mtc_group_id = mtc_group_id
        self.pre_time = int(pre_time)
        self.post_time = int(post_time)
        self.turn_time_after_maint = int(post_time) # 示例：使用post_time作为维护后的停站时间
        self.is_spare = is_spare == 'true'
        self.priority = int(priority)
        self.what_if = what_if == 'true'

        # 默认重置值 - 这些通常来自更详细的维护数据
        self.reset_flying_hours_to = 5000 # 示例
        self.reset_cycles_to = 1000     # 示例
        self.reset_time_to_next_maint_to = 5000 # 示例，单位分钟

    def __repr__(self):
        return f"Maintenance(ID={self.id}, Aircraft={self.aircraft_id}, Airport={self.required_airport})"

class Slot:
    """
    表示一个机场时间槽（起飞或降落）。
    """
    def __init__(self, slot_id, airport_id, type, start_time, end_time):
        self.id = slot_id
        self.airport_id = airport_id
        self.type = type # 'departure' 或 'arrival'
        self.start_time = parse_datetime(start_time)
        self.end_time = parse_datetime(end_time)

    def __repr__(self):
        return f"Slot(ID={self.id}, {self.airport_id}, {self.type}, {self.start_time.isoformat()}-{self.end_time.isoformat()})"

class ModelData:
    """
    聚合所有模型数据，包括航班、飞机、维护计划、时间槽和参数。
    """
    def __init__(self):
        self.flights = {} # 存储Flight对象，以leg_id为键
        self.aircrafts = {} # 存储Aircraft对象，以tail ID为键
        self.maintenance_plans = {} # 存储MaintenancePlan对象，以event_id为键
        self.slots = {} # 存储Slot对象，以slot_id为键
        self.parameters = {} # 存储从XML文件加载的参数

    def load_schedule_xml(self, filepath):
        """
        从schedule.xml文件加载航班数据并初始化飞机对象。
        """
        tree = ET.parse(filepath)
        root = tree.getroot()
        namespace = {'flt': 'http://generated.recoverymanager.sabre.com/flightSchedule'}
        
        # 收集独特的飞机（机尾号）及其初始状态
        initial_aircraft_states = {} # {tail: {'initial_location': ..., 'initial_time': ...}}

        for flt_info in root.findall('flt:fltInfo', namespace):
            leg_id = flt_info.find('flt:legID', namespace).text
            tail = flt_info.find('flt:tail', namespace).text
            dep_port = flt_info.find('flt:depPort', namespace).text
            dep_time = flt_info.find('flt:depTime', namespace).text

            # 创建Flight对象
            flight = Flight(
                leg_id=leg_id,
                seq_num=flt_info.find('flt:seqNum', namespace).text,
                dep_port=dep_port,
                arr_port=flt_info.find('flt:arrPort', namespace).text,
                dep_time=dep_time,
                arr_time=flt_info.find('flt:arrTime', namespace).text,
                pub_sched_dep_time=flt_info.find('flt:pubSchedDepTime', namespace).text,
                pub_sched_arr_time=flt_info.find('flt:pubSchedArrTime', namespace).text,
                turn_time=flt_info.find('flt:turnTime', namespace).text,
                block_time=flt_info.find('flt:blockTime', namespace).text,
                stage=flt_info.find('flt:stage', namespace).text,
                fleet=flt_info.find('flt:fleet', namespace).text,
                crew_rating=flt_info.find('flt:crewRating', namespace).text,
                avg_fare=flt_info.find('flt:avgFare', namespace).text,
                flt_dt=flt_info.find('flt:fltDt', namespace).text,
                flt_num=flt_info.find('flt:fltNum', namespace).text,
                eqp=flt_info.find('flt:eqp', namespace).text,
                suffix=flt_info.find('flt:suffix', namespace).text,
                airline=flt_info.find('flt:airline', namespace).text,
                dep_status=flt_info.find('flt:depStatus', namespace).text,
                arr_status=flt_info.find('flt:arrStatus', namespace).text,
                tail=tail,
                service=flt_info.find('flt:service', namespace).text,
                what_if=flt_info.find('flt:whatIf', namespace).text,
                delete_ind=flt_info.find('flt:deleteInd', namespace).text
            )
            self.flights[leg_id] = flight

            # 更新初始飞机状态：如果此航班的起飞时间更早，或者是一个新飞机
            if tail not in initial_aircraft_states:
                initial_aircraft_states[tail] = {'initial_location': dep_port, 'initial_time': flight.scheduled_dep_time}
            else:
                if flight.scheduled_dep_time < initial_aircraft_states[tail]['initial_time']:
                    initial_aircraft_states[tail]['initial_time'] = flight.scheduled_dep_time
                    initial_aircraft_states[tail]['initial_location'] = dep_port
        
        # 根据收集到的初始状态创建Aircraft对象
        for tail, data in initial_aircraft_states.items():
            self.aircrafts[tail] = Aircraft(
                aircraft_id=tail,
                initial_location=data['initial_location'],
                initial_time=data['initial_time']
            )

    def load_parameter_xml(self, filepath):
        """
        从parameter.xml文件加载参数。
        """
        tree = ET.parse(filepath)
        root = tree.getroot()
        namespace = {'par': 'http://generated.recoverymanager.sabre.com/Parameters'}

        for param_element in root:
            # 检查根元素的直接子元素（带有命名空间）
            tag_name = param_element.tag.replace(f"{{{namespace['par']}}}", '')
            value = param_element.text
            if value is not None:
                if 'time' in tag_name.lower() or 'threshold' in tag_name.lower():
                    try:
                        self.parameters[tag_name] = parse_datetime(value)
                    except ValueError:
                        self.parameters[tag_name] = value # 如果不是datetime，则保留为字符串
                elif value.lower() in ['true', 'false']:
                    self.parameters[tag_name] = (value.lower() == 'true')
                else:
                    try:
                        self.parameters[tag_name] = int(value)
                    except ValueError:
                        try:
                            self.parameters[tag_name] = float(value)
                        except ValueError:
                            self.parameters[tag_name] = value
        
        # 如果有currentTime参数，用它来设置飞机的初始时间
        current_time = self.parameters.get('currentTime')
        if current_time:
            for ac_id, ac_obj in self.aircrafts.items():
                # 只有当飞机最早的航班时间晚于currentTime时才更新
                if ac_obj.initial_time < current_time:
                    ac_obj.initial_time = current_time # 假设飞机在currentTime时位于其初始位置

    def load_maintenance_xml(self, filepath):
        """
        从maintenance.xml文件加载维护计划数据。
        """
        tree = ET.parse(filepath)
        root = tree.getroot()
        namespace = {'mtc': 'http://generated.recoverymanager.sabre.com/mtcSchedule'}

        for mtc_event in root.findall('mtc:mtcEvent', namespace):
            event_id = mtc_event.find('mtc:eventId', namespace).text
            maintenance = MaintenancePlan(
                event_id=event_id,
                tail=mtc_event.find('mtc:tail', namespace).text,
                port=mtc_event.find('mtc:port', namespace).text,
                event_status=mtc_event.find('mtc:eventStatus', namespace).text,
                st_time=mtc_event.find('mtc:stTime', namespace).text,
                en_time=mtc_event.find('mtc:enTime', namespace).text,
                mtc_group_id=mtc_event.find('mtc:mtcGroupId', namespace).text,
                pre_time=mtc_event.find('mtc:preTime', namespace).text,
                post_time=mtc_event.find('mtc:postTime', namespace).text,
                is_spare=mtc_event.find('mtc:isSpare', namespace).text,
                priority=mtc_event.find('mtc:priority', namespace).text,
                what_if=mtc_event.find('mtc:whatIf', namespace).text
            )
            self.maintenance_plans[event_id] = maintenance

    def load_adhoc_parameters_xml(self, filepath):
        """
        从AdhocParameters.xml文件加载临时参数。
        """
        tree = ET.parse(filepath)
        root = tree.getroot()
        namespace = {
            'par': 'http://generated.recoverymanager.sabre.com/Parameters',
            'inc': 'http://generated.recoverymanager.sabre.com/CommonDef'
        }

        for param_element in root.findall('par:parameter', namespace):
            name = param_element.find('inc:name', namespace).text
            value = param_element.find('inc:value', namespace).text
            if value is not None:
                if value.lower() in ['true', 'false']:
                    self.parameters[name] = (value.lower() == 'true')
                else:
                    try:
                        self.parameters[name] = int(value)
                    except ValueError:
                        try:
                            self.parameters[name] = float(value)
                        except ValueError:
                            self.parameters[name] = value

    def load_all_data(self, base_path):
        """
        加载所有XML文件中的数据。
        """
        self.load_schedule_xml(f"{base_path}/Schedule.xml")
        self.load_parameter_xml(f"{base_path}/Parameters.xml")
        self.load_maintenance_xml(f"{base_path}/Maintenance.xml")
        self.load_adhoc_parameters_xml(f"{base_path}/AdhocParameters.xml")

        # 示例：手动创建时间槽，如果实际项目中没有提供slots.xml
        # 直接传递字符串，让Slot的构造函数内部调用parse_datetime
        self.slots['SLOT_CDG_DEP_1'] = Slot('SLOT_CDG_DEP_1', 'CDG', 'departure',
                                            "2012-07-21T00:00:00.000Z", 
                                            "2012-07-25T23:59:00.000Z") 
        self.slots['SLOT_MEX_ARR_1'] = Slot('SLOT_MEX_ARR_1', 'MEX', 'arrival',
                                            "2012-07-21T00:00:00.000Z", 
                                            "2012-07-25T23:59:00.000Z") 


class Path:
    """
    表示一个生成的飞机路径。
    """
    def __init__(self, aircraft_id, sequence_of_flights_and_maintenance, operating_cost, flights_covered, maintenance_covered, slots_used):
        self.id = id(self) # 路径的唯一ID
        self.aircraft_id = aircraft_id
        self.sequence = sequence_of_flights_and_maintenance # (航班/维护对象, 实际起飞/开始时间, 实际降落/结束时间)的列表
        self.operating_cost = operating_cost # 修改：此operating_cost应包含航班的随机运营成本
        self.flights_covered = flights_covered # 覆盖的航班ID集合
        self.maintenance_covered = maintenance_covered # 覆盖的维护ID集合
        self.slots_used = slots_used # 字典: {时间槽ID: 使用次数}
        self.reduced_cost = 0 # 路径的缩减成本，在定价问题中计算

    def __repr__(self):
        return f"Path(AC={self.aircraft_id}, Cost={self.operating_cost:.2f}, RC={self.reduced_cost:.2f}, Flights={len(self.flights_covered)})"

class PricingProblem:
    """
    定价问题（受限资源最短路径问题）的求解器。
    它为每架飞机找到具有负缩减成本的路径。
    """
    def __init__(self, model_data):
        self.model_data = model_data
        self.inverted_flt_delay_penalty = model_data.parameters.get('invertedFltDelayPenalty', 200)
        self.inverted_flt_swap_penalty = model_data.parameters.get('invertedFltSwapPenalty', 501)
        self.label_id_counter = itertools.count() # 用于生成唯一的标签ID

    def solve_for_aircraft(self, a_id, dual_variables):
        """
        为特定飞机解决定价问题（标签设置算法）。
        查找具有负缩减成本的路径。
        """
        aircraft = self.model_data.aircrafts[a_id]
        flights = self.model_data.flights
        maintenance_plans = {m_id: plan for m_id, plan in self.model_data.maintenance_plans.items()
                             if plan.aircraft_id == a_id and plan.event_status == 'Planned'}
        slots = self.model_data.slots

        graph_nodes = {'start', 'end'}
        for f_id in flights:
            graph_nodes.add(f_id)
        
        adj_list = self._build_aircraft_graph(aircraft, flights)

        priority_queue = []
        # labels 结构更新：存储有效的标签列表。
        # 键是节点ID，值是一个字典，再次以 label_id 为键，存储实际的标签数据
        labels = {node: {} for node in graph_nodes} 

        # 初始化起始节点标签
        initial_rc = -dual_variables['gamma'].get(a_id, 0) # 飞机a_id的gamma对偶变量
        
        initial_label_id = next(self.label_id_counter)
        initial_label = {
            'label_id': initial_label_id, # 新增：唯一的标签ID
            'rc': initial_rc, # 缩减成本
            'delay': timedelta(minutes=0), # 累积延误
            'avail_flying_hours': aircraft.initial_available_flying_hours, # 可用飞行小时
            'remaining_cycles': aircraft.initial_remaining_cycles,       # 剩余循环数
            'time_to_next_maintenance': aircraft.initial_time_to_next_maintenance, # 距离下次维护时间，单位分钟
            'executed_maintenance': frozenset(), # 已执行的维护计划ID集合 (使用frozenset可哈希)
            'path_sequence': [], # (航班ID或维护ID, 实际起飞/开始时间, 实际降落/结束时间)列表
            'current_location': aircraft.initial_location, # 飞机当前位置
            'current_time': aircraft.initial_time, # 飞机当前可用时间
            'flights_covered_set': frozenset(), # 路径覆盖的航班ID集合
            'maintenance_covered_set': frozenset(), # 路径覆盖的维护ID集合
            'slots_used_dict': {}, # {时间槽ID: 使用次数}
            'is_valid': True # 标记标签是否仍然有效
        }
        labels['start'][initial_label_id] = initial_label
        # 优先队列中存储 (成本, 节点ID, 标签ID)
        heapq.heappush(priority_queue, (initial_rc, 'start', initial_label_id)) 

        generated_paths = []

        while priority_queue:
            current_rc, u_node_id, current_label_id = heapq.heappop(priority_queue)

            # 检查标签是否存在且仍然有效
            if current_label_id not in labels[u_node_id] or not labels[u_node_id][current_label_id]['is_valid']:
                continue # 此标签已被支配或移除，跳过

            current_label = labels[u_node_id][current_label_id]

            # 检查是否是旧的、劣质的路径
            if current_rc > current_label['rc']:
                continue
            
            if u_node_id == 'end':
                # 到达结束节点，找到潜在路径
                if current_label['rc'] < -1e-6: # 检查缩减成本是否显著为负
                    path_obj = self._construct_path_from_label(a_id, current_label)
                    path_obj.reduced_cost = current_label['rc']
                    generated_paths.append(path_obj)
                continue

            # 处理后继节点 (航班)
            successors = adj_list.get(u_node_id, [])
            for v_node_id in successors:
                new_label_for_flight = self._create_successor_label(current_label, u_node_id, v_node_id,
                                                        flights, maintenance_plans, slots, dual_variables,
                                                        a_id)

                if new_label_for_flight and self._check_resource_constraints(new_label_for_flight):
                    # 获取此节点v_node_id的现有有效标签
                    existing_valid_labels_at_v = list(labels[v_node_id].values())

                    # 应用支配规则：移除被new_label_for_flight支配的标签
                    dominated_by_new = [l_id for l_id, l_data in labels[v_node_id].items()
                                        if self._dominates(new_label_for_flight, l_data)]
                    for l_id in dominated_by_new:
                        labels[v_node_id][l_id]['is_valid'] = False # 标记为无效

                    # 检查new_label_for_flight是否被任何现有标签支配
                    if not self._is_dominated(new_label_for_flight, existing_valid_labels_at_v):
                        # 如果没有被支配，则添加新标签
                        new_label_id = next(self.label_id_counter)
                        new_label_for_flight['label_id'] = new_label_id
                        labels[v_node_id][new_label_id] = new_label_for_flight
                        heapq.heappush(priority_queue, (new_label_for_flight['rc'], v_node_id, new_label_id))
                    else:
                        new_label_for_flight['is_valid'] = False # 标记为无效

            # 考虑维护：在处理完航班节点后，但在转向下一个航班之前
            airport_for_maint_check = None
            if u_node_id in flights:
                airport_for_maint_check = flights[u_node_id].destination
            elif u_node_id == 'start':
                airport_for_maint_check = current_label['current_location']

            if airport_for_maint_check:
                maint_labels = self._consider_maintenance(current_label, airport_for_maint_check,
                                                          maintenance_plans, dual_variables)
                for ml in maint_labels:
                    if self._check_resource_constraints(ml):
                        # 如果进行了维护，则将新标签添加到当前节点，因为维护完成后飞机仍在该机场可用
                        existing_valid_labels_at_u = list(labels[u_node_id].values())

                        # 移除被ml支配的标签
                        dominated_by_ml = [l_id for l_id, l_data in labels[u_node_id].items()
                                            if self._dominates(ml, l_data)]
                        for l_id in dominated_by_ml:
                            labels[u_node_id][l_id]['is_valid'] = False # 标记为无效

                        # 标记ml是否被现有标签支配
                        if not self._is_dominated(ml, existing_valid_labels_at_u):
                            ml_id = next(self.label_id_counter)
                            ml['label_id'] = ml_id
                            labels[u_node_id][ml_id] = ml
                            heapq.heappush(priority_queue, (ml['rc'], u_node_id, ml_id)) # 维护后重新推入当前节点
                        else:
                            ml['is_valid'] = False # 标记为无效

        return generated_paths

    def _build_aircraft_graph(self, aircraft, flights):
        """
        为特定飞机构建有向无环图。
        节点包括'start'、'end'和所有相关航班。
        核心修改：PP现在考虑所有航班，而不仅仅是分配给当前飞机的航班，以允许飞机调换。
        """
        adj_list = {}
        # 考虑所有航班作为潜在节点
        all_flight_nodes = list(flights.keys())
        all_flight_nodes.sort(key=lambda f_id: flights[f_id].scheduled_dep_time)

        # 添加起始节点连接：任何航班都可以是起始航班，只要飞机可以到达其起飞机场
        for f_id in all_flight_nodes:
            flight = flights[f_id]
            # 飞机从其初始位置和时间出发，可以执飞任何匹配起点的航班
            # 不需要严格要求 flight.origin == aircraft.initial_location
            # 因为飞机可以先飞到其他机场 (这在 PP 中通过成本体现，但图的连接应该允许)
            # 简化的初始连接：所有航班都可能从'start'连接，但实际可行性会在标签扩展时检查
            adj_list.setdefault('start', []).append(f_id)


        # 构建航班之间的连接
        for i, u_f_id in enumerate(all_flight_nodes):
            flight_u = flights[u_f_id]
            for v_f_id in all_flight_nodes: 
                flight_v = flights[v_f_id]

                # 确保航班v的起飞时间晚于航班u的到达时间 (允许在机场停站)
                if flight_v.scheduled_dep_time >= flight_u.scheduled_arr_time + timedelta(minutes=flight_u.turn_time):
                    # 检查可行连接：航班u的到达机场必须是航班v的起飞机场
                    if flight_u.destination == flight_v.origin:
                        adj_list.setdefault(u_f_id, []).append(v_f_id)

        # 添加从最后一个可能航班到'end'节点的连接
        for f_id in all_flight_nodes:
            # 如果航班在图中没有后继，则它可能是终点航班
            if f_id not in adj_list: 
                adj_list.setdefault(f_id, []).append('end')
            # 此外，任何航班都可以直接连接到'end'，表示飞机在该航班后结束路径
            if 'end' not in adj_list.get(f_id, []):
                adj_list.setdefault(f_id, []).append('end')

        return adj_list

    def _create_successor_label(self, parent_label, u_node_id, v_node_id, flights, maintenance_plans, slots, dual_variables, aircraft_id):
        """
        基于父标签和到后继航班的连接创建新的标签。
        此函数更新成本、资源和路径序列。
        """
        new_label = parent_label.copy()
        new_label['path_sequence'] = parent_label['path_sequence'][:] # 深度复制列表
        new_label['flights_covered_set'] = parent_label['flights_covered_set'].copy()
        new_label['maintenance_covered_set'] = parent_label['maintenance_covered_set'].copy()
        new_label['slots_used_dict'] = parent_label['slots_used_dict'].copy()
        new_label['is_valid'] = True # 新标签默认为有效


        if v_node_id == 'end':
            # 处理结束节点逻辑，没有进一步操作
            return new_label

        flight_v = flights[v_node_id]

        # 检查此航班是否已被当前路径覆盖，避免重复
        if flight_v.leg_id in new_label['flights_covered_set']:
            return None # 已经覆盖，不能再次加入

        # 计算在u机场的实际到达时间，以及在v机场的起飞时间
        actual_prev_event_finish_time = parent_label['current_time'] # 飞机在当前位置准备就绪的时间

        # 确定航班v起飞前所需的停站时间
        turn_time_minutes = 0
        if len(parent_label['path_sequence']) > 0:
            last_item_id = parent_label['path_sequence'][-1][0]
            if last_item_id in maintenance_plans:
                turn_time_minutes = maintenance_plans[last_item_id].turn_time_after_maint
            elif last_item_id in flights:
                turn_time_minutes = flights[last_item_id].turn_time
        # 如果从'start'节点开始，则停站时间为0

        # 计算航班v最早的起飞时间
        earliest_dep_time_v = actual_prev_event_finish_time + timedelta(minutes=turn_time_minutes)

        # 检查飞机是否能在航班v的起飞机场（flight_v.origin）
        # 如果当前位置不匹配，需要考虑调机成本（隐式处理，或者假设 PP 允许调机，但会增加巨大的成本）
        # 在这里，我们假设飞机可以瞬间移动到起飞机场，但会产生巨大成本
        # 实际的更复杂模型会包含 Ferry Flight (调机航班)

        # 检查位置匹配。如果位置不匹配，直接返回None (不考虑此连接)
        # 只有当父节点是'start'时才允许飞机在初始位置不匹配航班起点，这意味着它可以从初始位置飞到航班起点
        # 但这里我们简化为：如果不是从start开始，则必须位置匹配
        if parent_label['current_location'] != flight_v.origin and u_node_id != 'start':
            return None # 不考虑非邻接机场的直接连接，除非是路径的起点


        # 计算实际起飞时间和产生的延误
        if earliest_dep_time_v > flight_v.scheduled_dep_time:
            delay_td = earliest_dep_time_v - flight_v.scheduled_dep_time
            actual_dep_time_v = earliest_dep_time_v
        else:
            delay_td = timedelta(minutes=0)
            actual_dep_time_v = flight_v.scheduled_dep_time
        
        actual_arr_time_v = actual_dep_time_v + flight_v.flight_time

        # --- 更新成本和对偶变量 ---
        C_a_swap_v = calculate_swap_cost(flight_v.tail, aircraft_id, self.inverted_flt_swap_penalty)
        # 核心修改：使用 pub_sched_dep_time 计算延误成本
        C_a_delay_v = calculate_delay_cost(actual_dep_time_v, flight_v.pub_sched_dep_time, self.inverted_flt_delay_penalty)
        
        # 将航班自身的运营成本也加入
        flight_own_op_cost = flight_v.operating_cost 

        # 计算时间槽成本
        slot_cost_v = 0
        slots_used_by_flight_v = {}
        for s_id, slot_obj in slots.items():
            # 检查起飞时间槽是否适用
            if slot_obj.airport_id == flight_v.origin and slot_obj.type == 'departure' and \
            slot_obj.start_time <= actual_dep_time_v <= slot_obj.end_time:
                slot_cost_v += dual_variables['theta'].get(s_id, 0)
                slots_used_by_flight_v[s_id] = slots_used_by_flight_v.get(s_id, 0) + 1
            # 检查到达时间槽是否适用
            if slot_obj.airport_id == flight_v.destination and slot_obj.type == 'arrival' and \
            slot_obj.start_time <= actual_arr_time_v <= slot_obj.end_time:
                slot_cost_v += dual_variables['theta'].get(s_id, 0)
                slots_used_by_flight_v[s_id] = slots_used_by_flight_v.get(s_id, 0) + 1
        
        # 本段的缩减成本计算
        # 航班v的运营成本：调换成本 + 延误成本 + 航班自身运营成本
        operating_cost_for_flight_v = C_a_swap_v + C_a_delay_v + flight_own_op_cost

        # 缩减成本项：(运营成本) - (航班覆盖对偶) - (所有使用时间槽的对偶之和)
        # 注意：这里的beta包含了RMP目标函数中y_f的惩罚，所以直接减去是正确的。
        rc_term_for_flight_v = operating_cost_for_flight_v - dual_variables['beta'].get(v_node_id, 0) - slot_cost_v
        new_label['rc'] += rc_term_for_flight_v
        new_label['delay'] += delay_td

        # --- 更新资源约束 (可用飞行小时, 剩余循环数, 距离下次维护时间) ---
        new_label['avail_flying_hours'] -= flight_v.flight_time.total_seconds() / 3600 # 将timedelta转换为小时
        new_label['remaining_cycles'] -= 1
        new_label['time_to_next_maintenance'] -= (flight_v.flight_time.total_seconds() / 60 + turn_time_minutes) # 总时间（分钟）

        # --- 更新路径序列和当前状态 ---
        new_label['path_sequence'].append((v_node_id, actual_dep_time_v, actual_arr_time_v))
        new_label['flights_covered_set'] = new_label['flights_covered_set'].union({v_node_id})
        for s_id, count in slots_used_by_flight_v.items():
            new_label['slots_used_dict'][s_id] = new_label['slots_used_dict'].get(s_id, 0) + count

        new_label['current_time'] = actual_arr_time_v # 飞机到达目的地
        new_label['current_location'] = flight_v.destination

        return new_label

    def _consider_maintenance(self, current_label, airport_for_maint_check, maintenance_plans, dual_variables):
        """
        考虑在当前机场执行维护。如果可行，则生成新的维护标签。
        """
        new_maint_labels = []

        # 检查适用于当前飞机位置且尚未执行的维护计划
        for m_id, m_plan in maintenance_plans.items():
            # 检查此维护是否已被当前路径覆盖，避免重复
            if m_id in current_label['maintenance_covered_set']:
                continue

            if m_plan.required_airport == airport_for_maint_check and m_plan.event_status == 'Planned':
                # 确保维护可以在其时间窗内，并在飞机当前到达时间之后执行
                maint_start_time = max(current_label['current_time'], m_plan.start_time)
                maint_end_time = maint_start_time + m_plan.duration

                if maint_end_time <= m_plan.end_time: # 确保维护在允许的时间窗内完成
                    maint_label = current_label.copy()
                    maint_label['path_sequence'] = current_label['path_sequence'][:]
                    maint_label['executed_maintenance'] = current_label['executed_maintenance'].union({m_id})
                    maint_label['maintenance_covered_set'] = current_label['maintenance_covered_set'].union({m_id})
                    maint_label['is_valid'] = True # 新标签默认为有效

                    # 调整维护的缩减成本（负值，因为它是一个“成本”扣除）
                    maint_cost_term = -dual_variables['delta'].get(m_id, 0)
                    maint_label['rc'] += maint_cost_term

                    # 维护后更新资源（重置/补充）
                    maint_label['avail_flying_hours'] = m_plan.reset_flying_hours_to
                    maint_label['remaining_cycles'] = m_plan.reset_cycles_to
                    maint_label['time_to_next_maintenance'] = m_plan.reset_time_to_next_maint_to

                    # 更新current_time以反映维护持续时间
                    maint_label['current_time'] = maint_end_time
                    maint_label['path_sequence'].append((m_id, maint_start_time, maint_end_time))
                    
                    new_maint_labels.append(maint_label)
        return new_maint_labels

    def _check_resource_constraints(self, label):
        """
        检查标签的资源约束是否满足。
        所有资源值必须为正（或至少非负）。
        """
        return label['avail_flying_hours'] >= 0 and \
            label['remaining_cycles'] >= 0 and \
            label['time_to_next_maintenance'] >= 0

    def _is_dominated(self, new_label, existing_labels_list):
        """
        检查新标签是否被任何现有标签支配。
        支配规则：如果一个现有标签在所有成本（缩减成本、延误）上都优于或等于新标签，
        并且在所有资源（可用飞行小时、剩余循环、距下次维护时间）上都优于或等于新标签，则支配新标签。
        已执行的维护集合不直接参与支配检查。
        """
        for existing_l in existing_labels_list:
            if not existing_l['is_valid']: # 只与有效标签进行支配检查
                continue
            if existing_l['rc'] <= new_label['rc'] and \
            existing_l['delay'] <= new_label['delay'] and \
            existing_l['avail_flying_hours'] >= new_label['avail_flying_hours'] and \
            existing_l['remaining_cycles'] >= new_label['remaining_cycles'] and \
            existing_l['time_to_next_maintenance'] >= new_label['time_to_next_maintenance'] and \
            existing_l['flights_covered_set'].issuperset(new_label['flights_covered_set']) and \
            existing_l['maintenance_covered_set'].issuperset(new_label['maintenance_covered_set']):
                return True
        return False

    def _dominates(self, label1, label2):
        """
        检查label1是否严格支配label2。
        严格支配意味着在所有条件上都优于或等于，并且至少在一个条件上严格优于。
        """
        if not label1['is_valid'] or not label2['is_valid']: # 只有有效的标签才能支配
            return False

        conditions_met = (
            label1['rc'] <= label2['rc'] and
            label1['delay'] <= label2['delay'] and
            label1['avail_flying_hours'] >= label2['avail_flying_hours'] and
            label1['remaining_cycles'] >= label2['remaining_cycles'] and
            label1['time_to_next_maintenance'] >= label2['time_to_next_maintenance'] and
            label1['flights_covered_set'].issuperset(label2['flights_covered_set']) and
            label1['maintenance_covered_set'].issuperset(label2['maintenance_covered_set'])
        )
        strict_condition_met = (
            label1['rc'] < label2['rc'] or
            label1['delay'] < label2['delay'] or
            label1['avail_flying_hours'] > label2['avail_flying_hours'] or
            label1['remaining_cycles'] > label2['remaining_cycles'] or
            label1['time_to_next_maintenance'] > label2['time_to_next_maintenance'] or
            len(label1['flights_covered_set']) > len(label2['flights_covered_set']) or
            len(label1['maintenance_covered_set']) > len(label2['maintenance_covered_set'])
        )
        return conditions_met and strict_condition_met

    def _construct_path_from_label(self, aircraft_id, label):
        """
        根据标签的路径序列和其他详细信息构建Path对象。
        计算路径的实际运营成本（不含对偶变量）。
        """
        total_op_cost = 0
        flights_in_path = set()
        maintenance_in_path = set()
        slots_used_in_path = {}

        # 遍历标签的路径序列，计算总运营成本，并收集覆盖的航班、维护和时间槽。
        for i, (item_id, actual_dep_time, actual_arr_time) in enumerate(label['path_sequence']):
            if item_id in self.model_data.flights:
                flight = self.model_data.flights[item_id]
                flights_in_path.add(item_id)

                # 重新计算此航班段的延误成本，使用 pub_sched_dep_time
                delay_cost = calculate_delay_cost(actual_dep_time, flight.pub_sched_dep_time, self.inverted_flt_delay_penalty)
                
                # 调换成本（仅当航班的原始机尾号与当前飞机ID不同时产生）
                swap_cost = calculate_swap_cost(flight.tail, aircraft_id, self.inverted_flt_swap_penalty)
                
                # 添加航班自身的运营成本
                flight_own_op_cost = flight.operating_cost
                
                total_op_cost += delay_cost + swap_cost + flight_own_op_cost

                # 收集时间槽使用情况（根据实际时间重新评估）
                for s_id, slot_obj in self.model_data.slots.items():
                    if (slot_obj.airport_id == flight.origin and slot_obj.type == 'departure' and
                        slot_obj.start_time <= actual_dep_time <= slot_obj.end_time) or \
                    (slot_obj.airport_id == flight.destination and slot_obj.type == 'arrival' and
                        slot_obj.start_time <= actual_arr_time <= slot_obj.end_time):
                        slots_used_in_path[s_id] = slots_used_in_path.get(s_id, 0) + 1

            elif item_id in self.model_data.maintenance_plans:
                maintenance_in_path.add(item_id)
                # 维护本身在此模型运营成本中没有直接的“运营成本”，
                # 它的成本通过对偶变量调整和资源重置来管理。
                
        return Path(aircraft_id, label['path_sequence'], total_op_cost, flights_in_path, maintenance_in_path, slots_used_in_path)

class RestrictedMasterProblem:
    """
    受限主问题（RMP）的构建和求解器。
    RMP是一个线性规划问题，它选择一组路径来覆盖所有航班和维护，同时最小化总成本。
    使用 Gurobi 求解器。
    """
    def __init__(self, model_data):
        self.model_data = model_data
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0) # 静默Gurobi输出
        self.env.start()
        self.model = gp.Model("AircraftRecoveryRMP", env=self.env)
        self.model.ModelSense = GRB.MINIMIZE
        self.slot_capacities = {s_id: 10 for s_id in model_data.slots.keys()} # 假设默认容量为10

        # 决策变量:
        # x_apr: 如果飞机a选择路径p，则为1
        self.x = {} # 存储Gurobi变量
        # y_f: 如果航班f被取消，则为1
        self.y = self.model.addVars(model_data.flights.keys(), vtype=GRB.CONTINUOUS, name="FlightCancelled", lb=0, ub=1)
        # z_m: 如果维护m被取消，则为1
        self.z = self.model.addVars(model_data.maintenance_plans.keys(), vtype=GRB.CONTINUOUS, name="MaintenanceCancelled", lb=0, ub=1)

        # 存储添加的路径
        self.all_paths = {a_id: [] for a_id in model_data.aircrafts}

        # 对偶变量的存储 (由Gurobi填充)
        self.beta = {}  # 航班覆盖约束的对偶
        self.gamma = {} # 飞机使用约束的对偶
        self.delta = {} # 维护覆盖约束的对偶
        self.theta = {} # 时间槽约束的对偶

        # 初始化目标函数 - 仅包含取消成本部分
        self.inverted_flt_unassign_penalty = model_data.parameters.get('invertedFltUnassignPenalty', 800)
        self.inverted_mtc_unassign_penalty = model_data.parameters.get('invertedMtcUnassignPenalty', 500)
        
        obj_expr = gp.quicksum(self.inverted_flt_unassign_penalty * self.y[f_id] for f_id in self.model_data.flights.keys())
        obj_expr += gp.quicksum(self.inverted_mtc_unassign_penalty * self.z[m_id] for m_id in self.model_data.maintenance_plans.keys())
        self.model.setObjective(obj_expr)

        # 初始化约束
        self._initialize_constraints()

    def _initialize_constraints(self):
        """
        初始化RMP中的基本约束。
        """
        # 1. 每架飞机只能选择一条路径
        self.aircraft_path_constraints = {}
        for a_id in self.model_data.aircrafts:
            # Gurobi的addConstr需要一个线性表达式
            self.aircraft_path_constraints[a_id] = self.model.addConstr(
                gp.LinExpr() <= 1, name=f"AircraftPathChoice_{a_id}"
            )

        # 2. 每个航班必须被恰好一条路径覆盖，或被取消
        self.flight_coverage_constraints = {}
        for f_id in self.model_data.flights:
            # 初始时，左侧为0，右侧为 self.y[f_id]
            # 后续添加路径时，会添加 x_apr 到左侧
            self.flight_coverage_constraints[f_id] = self.model.addConstr(
                self.y[f_id] == 1, name=f"FlightCoverage_{f_id}"
            )

        # 3. 每个维护事件必须被恰好一条路径覆盖，或被取消
        self.maintenance_coverage_constraints = {}
        for m_id in self.model_data.maintenance_plans:
            self.maintenance_coverage_constraints[m_id] = self.model.addConstr(
                self.z[m_id] == 1, name=f"MaintenanceCoverage_{m_id}"
            )

        # 4. 时间槽容量约束（如果有的话，每个时间槽最多使用N次）
        self.slot_capacity_constraints = {}
        for s_id in self.model_data.slots:
            self.slot_capacity_constraints[s_id] = self.model.addConstr(
                gp.LinExpr() <= 10, name=f"SlotCapacity_{s_id}" # 初始时左侧为0
            )
        self.model.update()

    def add_path(self, path: Path):
        """
        将新生成的路径作为新列添加到RMP中。
        更新目标函数和相关约束。
        """
        a_id = path.aircraft_id
        path_id = path.id

        # 确保路径ID是唯一的
        if (a_id, path_id) in self.x:
            print(f"警告: 路径 {(a_id, path_id)} 已存在。跳过添加。")
            return

        # 1. 为新路径创建决策变量
        self.x[(a_id, path_id)] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"x_{a_id}_{path_id}", lb=0, ub=1)
        self.all_paths[a_id].append(path)

        # 2. 更新目标函数：添加路径的运营成本
        self.model.setObjective(self.model.getObjective() + path.operating_cost * self.x[(a_id, path_id)])

        # --- 3. 更新飞机路径选择约束 (AircraftPathChoice) ---
        # 针对每个飞机，所有分配给它的路径只能被选择一次 (或总和为1)
        self.model.remove(self.aircraft_path_constraints[a_id]) # 首先移除旧的约束
        aircraft_path_lhs = LinExpr()
        for p in self.all_paths[a_id]:
            aircraft_path_lhs.add(self.x[(p.aircraft_id, p.id)])
        self.aircraft_path_constraints[a_id] = self.model.addConstr(
            aircraft_path_lhs <= 1,
            name=f"AircraftPathChoice_{a_id}"
        )

        # --- 4. 更新航班覆盖约束 (FlightCoverage) ---
        # 对于新路径覆盖的每个航班，都需要更新其对应的航班覆盖约束
        for f_id in path.flights_covered:
            self.model.remove(self.flight_coverage_constraints[f_id]) # 移除旧约束

            flight_coverage_lhs = LinExpr()
            # 遍历所有飞机及其所有路径，找出哪些路径覆盖了当前的航班 f_id
            for current_a_id in self.all_paths: 
                for p in self.all_paths[current_a_id]: 
                    if f_id in p.flights_covered: 
                        flight_coverage_lhs.add(self.x[(p.aircraft_id, p.id)]) 
            # 添加新的约束：所有覆盖航班 f_id 的路径变量之和 + y_f == 1
            self.flight_coverage_constraints[f_id] = self.model.addConstr(
                flight_coverage_lhs + self.y[f_id] == 1,
                name=f"FlightCoverage_{f_id}"
            )

        # --- 5. 更新维护覆盖约束 (MaintenanceCoverage) ---
        for m_id in path.maintenance_covered:
            self.model.remove(self.maintenance_coverage_constraints[m_id]) # 移除旧约束

            maintenance_coverage_lhs = LinExpr()
            for current_a_id in self.all_paths:
                for p in self.all_paths[current_a_id]:
                    if m_id in p.maintenance_covered:
                        maintenance_coverage_lhs.add(self.x[(p.aircraft_id, p.id)])
            # 添加新的约束：所有覆盖维护任务 m_id 的路径变量之和 + z_m == 1
            self.maintenance_coverage_constraints[m_id] = self.model.addConstr(
                maintenance_coverage_lhs + self.z[m_id] == 1,
                name=f"MaintenanceCoverage_{m_id}"
            )

        # --- 6. 更新时间槽容量约束 (SlotCapacity) ---
        for s_id, _ in path.slots_used.items():
            if s_id not in self.slot_capacities:
                self.slot_capacities[s_id] = 10 
        for s_id, _ in path.slots_used.items():
            self.model.remove(self.slot_capacity_constraints[s_id]) # 移除旧约束

            slot_capacity_lhs = LinExpr()
            for current_a_id in self.all_paths:
                for p in self.all_paths[current_a_id]:
                    if s_id in p.slots_used:
                        slot_capacity_lhs.add(p.slots_used[s_id] * self.x[(p.aircraft_id, p.id)])
            self.slot_capacity_constraints[s_id] = self.model.addConstr(
                slot_capacity_lhs <= self.slot_capacities[s_id], 
                name=f"SlotCapacity_{s_id}"
            )
        self.model.update()

    def solve(self):
        """
        求解RMP（LP松弛）。
        提取对偶变量。
        """
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            print(f"RMP求解失败，状态: {self.model.Status} - {GRB.Status[self.model.Status]}")
            return False

        # 提取对偶变量
        # 航班覆盖约束的对偶变量现在对应 (sum(x_apr) + y_f == 1)
        self.beta = {f_id: self.flight_coverage_constraints[f_id].Pi for f_id in self.model_data.flights}
        self.gamma = {a_id: self.aircraft_path_constraints[a_id].Pi for a_id in self.model_data.aircrafts}
        self.delta = {m_id: self.maintenance_coverage_constraints[m_id].Pi for m_id in self.model_data.maintenance_plans}
        self.theta = {s_id: self.slot_capacity_constraints[s_id].Pi for s_id in self.model_data.slots}
        
        print(f"RMP LP松弛最优值: {self.model.ObjVal:.2f}")
        return True


class ColumnGenerationSolver:
    """
    列生成算法的实现，用于解决飞机恢复问题。
    它迭代地解决RMP（主问题）和PP（子问题），直到没有新的负缩减成本路径。
    """
    def __init__(self, model_data):
        self.model_data = model_data
        self.rmp_solver = RestrictedMasterProblem(model_data)
        self.pp_solver = PricingProblem(model_data)

    def solve(self):
        # 步骤1：生成初始可行解
        # 初始解的目标是为每架飞机找到一条基础路径，并识别出那些在初始阶段无法被任何飞机覆盖的航班/维护，将其作为“初始取消”。
        # 这些“初始取消”在RMP中不会被固定为1，而是作为决策变量，允许后续PP找到的路径来覆盖它们。
        initial_paths, initially_uncovered_flights, initially_uncovered_maintenance = self._generate_initial_solutions()
        
        if not initial_paths and (not initially_uncovered_flights and not initially_uncovered_maintenance):
            print("未能生成初始路径或确定任何需取消的事件。终止列生成。")
            return

        for path in initial_paths:
            self.rmp_solver.add_path(path)

        # 确保RMP中的所有航班和维护覆盖约束都已更新
        # 这里需要特别处理那些未被任何初始路径覆盖的航班和维护，
        # 它们的覆盖约束应该仍然是 `y_f == 1` 或 `z_m == 1`
        # （因为没有路径覆盖它们，所以只能被取消），
        # RMP的 `add_path` 已经处理了路径覆盖的情况。
        # 我们可以确保对于未覆盖的航班，约束仍然是 `y_f == 1`。

        self.rmp_solver.model.update()

        # 解决RMP一次以初始化对偶变量
        self.rmp_solver.solve() 

        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- 列生成迭代 {iteration} ---")

            # 步骤2：解决受限主问题（RMP）的LP松弛
            rmp_solved = self.rmp_solver.solve()
            if not rmp_solved:
                print("RMP求解失败。终止。")
                break

            dual_vars = {
                'beta': self.rmp_solver.beta,
                'gamma': self.rmp_solver.gamma,
                'delta': self.rmp_solver.delta,
                'theta': self.rmp_solver.theta
            }

            # 步骤3：为每架飞机解决定价问题
            new_paths_found = []
            min_reduced_cost_overall = 0 # 跟踪所有PP中最小的缩减成本

            for a_id in self.model_data.aircrafts:
                paths_for_aircraft = self.pp_solver.solve_for_aircraft(a_id, dual_vars)
                for path in paths_for_aircraft:
                    new_paths_found.append(path)
                    min_reduced_cost_overall = min(min_reduced_cost_overall, path.reduced_cost) 

            if not new_paths_found or min_reduced_cost_overall >= -1e-6: # 检查是否存在新的负缩减成本路径
                print("未找到新的负缩减成本路径。终止列生成。")
                break

            # 步骤4：将新路径（列）添加到RMP
            for path in new_paths_found:
                self.rmp_solver.add_path(path)
                print(f"  为飞机 {path.aircraft_id} 添加新路径，缩减成本 {path.reduced_cost:.4f}")

        # 列生成结束后，将RMP作为整数规划（IP）求解
        print("\n--- 最终IP求解 ---")
        # 将路径变量从连续变量改为二值变量
        for (a_id, path_id), var in self.rmp_solver.x.items():
            var.vtype = GRB.BINARY
        for f_id, var in self.rmp_solver.y.items():
            var.vtype = GRB.BINARY
        for m_id, var in self.rmp_solver.z.items(): 
            var.vtype = GRB.BINARY
        self.rmp_solver.model.update() 
        
        self.rmp_solver.model.optimize() 

        if self.rmp_solver.model.Status == GRB.OPTIMAL:
            print("\n找到飞机恢复问题的最优解:")
            print(f"总成本: {self.rmp_solver.model.ObjVal:.2f}")
            self._report_solution()
        else:
            print(f"\n未能找到最优整数解，状态: {self.rmp_solver.model.Status} - {GRB.Status[self.rmp_solver.model.Status]}")


    def _generate_initial_solutions(self):
        """
        生成初始可行解。
        策略：为每架飞机构建一条尽可能长的、可行的（允许延误但无严重冲突）路径。
        任何未被这些初始路径覆盖的航班/维护，都将被视为“初始未覆盖”，
        让RMP通过其取消变量（y_f, z_m）来处理，或等待PP生成新列来覆盖。
        """
        initial_paths = []
        # 用于跟踪在任何初始路径中已被覆盖的航班和维护
        covered_flights_in_initial_paths = set()
        covered_maintenance_in_initial_paths = set()

        print("正在生成初始解...")

        # 整理每架飞机的原始计划事件（航班和维护）
        aircraft_original_events = {ac_id: [] for ac_id in self.model_data.aircrafts}
        for f_id, flight in self.model_data.flights.items():
            # 即使是中断航班，也先放入其原始飞机计划中，尝试恢复
            aircraft_original_events[flight.tail].append(flight)
        for m_id, maint in self.model_data.maintenance_plans.items():
            if maint.aircraft_id in aircraft_original_events:
                aircraft_original_events[maint.aircraft_id].append(maint)
        
        # 对每个飞机的事件进行排序，以便按时间顺序构建路径
        for ac_id in aircraft_original_events:
            aircraft_original_events[ac_id].sort(key=lambda x: x.scheduled_dep_time if isinstance(x, Flight) else x.start_time)

        for ac_id, events_for_ac in aircraft_original_events.items():
            aircraft_obj = self.model_data.aircrafts[ac_id]
            
            current_time_for_path = aircraft_obj.initial_time
            current_location_for_path = aircraft_obj.initial_location
            current_path_op_cost = 0
            current_path_flights = set()
            current_path_maintenance = set()
            current_path_sequence = []
            
            # 用于尝试构建路径时的资源追踪
            current_avail_flying_hours = aircraft_obj.initial_available_flying_hours
            current_remaining_cycles = aircraft_obj.initial_remaining_cycles
            current_time_to_next_maintenance = aircraft_obj.initial_time_to_next_maintenance

            print(f"  为飞机 {ac_id} 尝试构建初始路径...")

            for event in events_for_ac:
                item_id = event.leg_id if isinstance(event, Flight) else event.id

                # 如果这个航班/维护已经被其他飞机的初始路径覆盖，则跳过
                if item_id in covered_flights_in_initial_paths or item_id in covered_maintenance_in_initial_paths:
                    continue

                event_start_time = event.scheduled_dep_time if isinstance(event, Flight) else event.start_time
                # event_end_time = event.scheduled_arr_time if isinstance(event, Flight) else event.end_time # 未直接使用，但用于参考

                # 计算停站时间（如果前一个事件是维护，则使用维护后的停站时间）
                turn_time_minutes = 0
                if current_path_sequence:
                    last_item_in_seq_id = current_path_sequence[-1][0]
                    if last_item_in_seq_id in self.model_data.flights:
                        turn_time_minutes = self.model_data.flights[last_item_in_seq_id].turn_time
                    elif last_item_in_seq_id in self.model_data.maintenance_plans:
                        turn_time_minutes = self.model_data.maintenance_plans[last_item_in_seq_id].turn_time_after_maint

                # 计算飞机最早可在当前位置执行下一个事件的时间
                earliest_event_ready_time = current_time_for_path + timedelta(minutes=turn_time_minutes)

                # 检查位置匹配
                location_match = False
                if isinstance(event, Flight) and event.origin == current_location_for_path:
                    location_match = True
                elif isinstance(event, MaintenancePlan) and event.required_airport == current_location_for_path:
                    location_match = True
                elif len(current_path_sequence) == 0 and event_start_time >= aircraft_obj.initial_time and event.origin == aircraft_obj.initial_location: # 初始航班/维护
                    location_match = True
                
                if not location_match:
                    # 如果位置不匹配，这个事件不能直接被当前飞机执行。
                    # 在初始解中，我们不会尝试复杂的调机，所以直接跳过。
                    # 后续RMP/PP会找到更好的策略。
                    continue

                # 尝试将事件加入路径
                if isinstance(event, Flight):
                    # 检查航班是否本身就中断，如果中断，在初始解中我们可能倾向于取消它，
                    # 但在这里我们还是尝试“强制”执行，让RPP来找到最佳解
                    
                    actual_dep_time = max(event.scheduled_dep_time, earliest_event_ready_time)
                    actual_arr_time = actual_dep_time + event.flight_time

                    # 检查资源约束
                    temp_flying_hours = current_avail_flying_hours - (event.flight_time.total_seconds() / 3600)
                    temp_cycles = current_remaining_cycles - 1
                    temp_maint_time = current_time_to_next_maintenance - (event.flight_time.total_seconds() / 60 + turn_time_minutes)
                    
                    if temp_flying_hours >= 0 and temp_cycles >= 0 and temp_maint_time >= 0:
                        # 资源足够，添加航班
                        # 核心修改：使用 pub_sched_dep_time 计算延误成本
                        delay_cost = calculate_delay_cost(actual_dep_time, event.pub_sched_dep_time, self.pp_solver.inverted_flt_delay_penalty)
                        swap_cost = calculate_swap_cost(event.tail, ac_id, self.pp_solver.inverted_flt_swap_penalty)
                        
                        current_path_op_cost += delay_cost + swap_cost + event.operating_cost
                        current_path_sequence.append((item_id, actual_dep_time, actual_arr_time))
                        current_path_flights.add(item_id)
                        
                        current_time_for_path = actual_arr_time
                        current_location_for_path = event.destination
                        current_avail_flying_hours = temp_flying_hours
                        current_remaining_cycles = temp_cycles
                        current_time_to_next_maintenance = temp_maint_time
                    else:
                        # 资源不足，无法执行后续航班
                        break 

                elif isinstance(event, MaintenancePlan):
                    if event.event_status not in ['Planned', 'COMPLETED']:
                        # 维护状态不正常，跳过。
                        continue

                    maint_start_time = max(event.start_time, earliest_event_ready_time)
                    maint_end_time = maint_start_time + event.duration

                    if maint_end_time <= event.end_time: # 维护必须在规定时间窗内完成
                        # 维护会重置资源，所以不需要检查之前的资源是否够用，只需更新
                        current_path_sequence.append((item_id, maint_start_time, maint_end_time))
                        current_path_maintenance.add(item_id)

                        current_time_for_path = maint_end_time
                        current_location_for_path = event.required_airport
                        current_avail_flying_hours = event.reset_flying_hours_to
                        current_remaining_cycles = event.reset_cycles_to
                        current_time_to_next_maintenance = event.reset_time_to_next_maint_to
                    else:
                        # 维护无法在规定时间窗内完成
                        continue # 继续尝试下一个事件
            
            if current_path_sequence: # 如果为这架飞机构建了任何路径
                new_path = Path(
                    aircraft_id=ac_id,
                    sequence_of_flights_and_maintenance=current_path_sequence,
                    operating_cost=current_path_op_cost,
                    flights_covered=frozenset(current_path_flights),
                    maintenance_covered=frozenset(current_path_maintenance),
                    slots_used={} # 初始解暂时不计算slots_used
                )
                initial_paths.append(new_path)
                covered_flights_in_initial_paths.update(current_path_flights)
                covered_maintenance_in_initial_paths.update(current_path_maintenance)
                print(f"  飞机 {ac_id} 的初始路径已构建 (覆盖航班数: {len(current_path_flights)}, 维护数: {len(current_path_maintenance)})。")
            else:
                print(f"  飞机 {ac_id} 未能构建任何初始路径。")
                # 对于未能构建路径的飞机，添加一个空路径，表示它虽然存在，但初始不执行任何任务
                empty_path = Path(
                    aircraft_id=ac_id,
                    sequence_of_flights_and_maintenance=[],
                    operating_cost=0,
                    flights_covered=frozenset(),
                    maintenance_covered=frozenset(),
                    slots_used={}
                )
                initial_paths.append(empty_path)
        
        # 确定在任何初始路径中都未被覆盖的航班和维护
        initially_uncovered_flights = set(self.model_data.flights.keys()) - covered_flights_in_initial_paths
        initially_uncovered_maintenance = set(self.model_data.maintenance_plans.keys()) - covered_maintenance_in_initial_paths

        print(f"生成了 {len(initial_paths)} 条初始路径。")
        print(f"  初始未被覆盖的航班: {len(initially_uncovered_flights)} 个")
        print(f"  初始未被覆盖的维护: {len(initially_uncovered_maintenance)} 个")
        
        return initial_paths, initially_uncovered_flights, initially_uncovered_maintenance


    def _report_solution(self):
        """
        详细报告找到的解决方案：选择的路径、取消的航班和维护，以及总成本。
        """
        print("\n--- 解决方案详情 ---")
        total_cost = self.rmp_solver.model.ObjVal
        print(f"总成本: {total_cost:.2f}")

        # 报告被选择的路径
        print("\n选择的飞机路径:")
        paths_selected_count = 0
        for a_id in self.rmp_solver.all_paths:
            for path in self.rmp_solver.all_paths[a_id]:
                if (a_id, path.id) in self.rmp_solver.x and self.rmp_solver.x[(a_id, path.id)].X > 0.5: # 如果路径变量为1
                    paths_selected_count += 1
                    print(f"  飞机 {a_id} 选择路径 ID {path.id} (运营成本: {path.operating_cost:.2f})")
                    for item, dep, arr in path.sequence:
                        if item in self.model_data.flights:
                            flight_obj = self.model_data.flights[item]
                            # 核心修改：使用 pub_sched_dep_time 报告延误成本
                            delay_cost = calculate_delay_cost(dep, flight_obj.pub_sched_dep_time, self.pp_solver.inverted_flt_delay_penalty)
                            swap_cost = calculate_swap_cost(flight_obj.tail, a_id, self.pp_solver.inverted_flt_swap_penalty)
                            print(f"    - 航班 {item} (始发: {flight_obj.origin}, 目的: {flight_obj.destination}, 实际起飞: {dep.isoformat()}, 实际降落: {arr.isoformat()}, 航班自身运营成本: {flight_obj.operating_cost}, 延误成本: {delay_cost:.2f}, 调换成本: {swap_cost:.2f})")
                        elif item in self.model_data.maintenance_plans:
                            print(f"    - 维护 {item} (机场: {self.model_data.maintenance_plans[item].required_airport}, 开始: {dep.isoformat()}, 结束: {arr.isoformat()})")
        if paths_selected_count == 0:
            print("  没有选择任何飞机路径 (这可能表示所有航班都被取消)。")

        # 报告被取消的航班
        print("\n取消的航班:")
        cancelled_flights_count = 0
        for f_id in self.rmp_solver.y:
            if self.rmp_solver.y[f_id].X > 0.5:
                cancelled_flights_count += 1
                flight = self.model_data.flights[f_id]
                print(f"  航班 {f_id} (从 {flight.origin} 到 {flight.destination}, 计划起飞: {flight.scheduled_dep_time.isoformat()}, 基准起飞: {flight.pub_sched_dep_time.isoformat()}, 原因: {flight.disruption_reason}, 取消惩罚: {self.rmp_solver.inverted_flt_unassign_penalty})")
        if cancelled_flights_count == 0:
            print("  没有航班被取消。")

        # 报告被取消的维护
        print("\n取消的维护事件:")
        cancelled_maint_count = 0
        for m_id in self.rmp_solver.z:
            if self.rmp_solver.z[m_id].X > 0.5:
                cancelled_maint_count += 1
                maint_plan = self.model_data.maintenance_plans[m_id]
                print(f"  维护 {m_id} (飞机: {maint_plan.aircraft_id}, 机场: {maint_plan.required_airport}, 计划开始: {maint_plan.start_time.isoformat()}, 状态: {maint_plan.event_status}, 取消惩罚: {self.rmp_solver.inverted_mtc_unassign_penalty})")
        if cancelled_maint_count == 0:
            print("  没有维护事件被取消。")

# --- 示例用法 ---
if __name__ == "__main__":
    model_data = ModelData()
    # 请根据您的实际路径修改这里
    base_path = "D:/software/Engines/Rmops/rmops_rt/scenarios/AM2/753666/input" 
    model_data.load_all_data(base_path)

    print(f"加载了 {len(model_data.flights)} 个航班。")
    print(f"加载了 {len(model_data.aircrafts)} 架飞机。")
    print(f"加载了 {len(model_data.maintenance_plans)} 个维护计划。")
    print(f"加载了 {len(model_data.slots)} 个时间槽（示例）。")
    print(f"参数: {model_data.parameters}")

    # 打印一些航班的随机运营成本以验证
    print("\n部分航班的随机运营成本:")
    for i, flight_id in enumerate(list(model_data.flights.keys())[:5]):
        flight = model_data.flights[flight_id]
        print(f"  航班 {flight.leg_id}: 运营成本 = {flight.operating_cost}")

    # 实例化列生成求解器并运行
    solver = ColumnGenerationSolver(model_data)
    solver.solve()
