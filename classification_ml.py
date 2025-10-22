"""
分类机器学习算法训练器
实现论文3.3.5节的分类算法训练
支持 SVM, LR, RF 等多种分类算法
"""
import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Set
from pathlib import Path
from collections import defaultdict

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from column_generation import ColumnGenerationSolver
from LambdaMart_train_1 import compute_features,ScenarioLoader


class ClassificationLabelStrategy:
    """
    分类标注策略(论文3.3.5节)
    标注规则:
    - 与目标不正常航班原飞机发生交换的飞机 -> A1 -> 标注为1
    - 与A1中飞机发生交换的飞机 -> A2 -> 标注为1
    - 继续扩展...直到所有直接或间接交换的飞机都标注为1
    - 其余飞机标注为0
    """
    
    def __init__(self):
        pass
    
    def label_aircraft_binary(self, 
                             optimal_solution: Dict,
                             disrupted_flight_id: int,
                             original_aircraft_id: str,
                             all_aircraft: Set[str]) -> Dict[str, int]:
        """
        对候选飞机进行二分类标注
        
        Args:
            optimal_solution: 列生成最优解
            disrupted_flight_id: 目标不正常航班ID
            original_aircraft_id: 原计划执飞的飞机ID
            all_aircraft: 所有候选飞机ID集合
            
        Returns:
            {aircraft_id: label} 字典, label为0或1
        """
        # 从最优解中提取路径和飞机交换关系
        paths = optimal_solution.get('paths', [])
        
        # 构建交换关系图
        swap_graph = self._build_swap_graph(paths, disrupted_flight_id, 
                                            original_aircraft_id)
        
        # 初始化标注(默认为0)
        labels = {ac: 0 for ac in all_aircraft}
        labeled_as_one = set()
        
        # ===== 第1层: 与原飞机发生交换的飞机(A1) =====
        if original_aircraft_id in swap_graph:
            A1 = swap_graph[original_aircraft_id]
            for ac in A1:
                if ac in all_aircraft:
                    labels[ac] = 1
                    labeled_as_one.add(ac)
            print(f"    A1 (标注1): {A1}")
        else:
            print(f"    A1: 空集")
        
        # ===== 逐层扩展: A2, A3, ... =====
        current_layer = labeled_as_one.copy()
        layer_num = 2
        
        while current_layer and layer_num <= 20:  # 最多扩展20层
            next_layer = set()
            
            for aircraft in current_layer:
                if aircraft in swap_graph:
                    for swap_aircraft in swap_graph[aircraft]:
                        if (swap_aircraft in all_aircraft and 
                            swap_aircraft not in labeled_as_one):
                            labels[swap_aircraft] = 1
                            labeled_as_one.add(swap_aircraft)
                            next_layer.add(swap_aircraft)
            
            if next_layer:
                print(f"    A{layer_num} (标注1): {next_layer}")
            
            current_layer = next_layer
            layer_num += 1
        
        # 统计
        num_positive = sum(1 for v in labels.values() if v == 1)
        num_negative = len(labels) - num_positive
        print(f"    标注统计: 正样本={num_positive}, 负样本={num_negative}")
        
        return labels
    
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


class ClassificationTrainer:
    """
    分类机器学习算法训练器
    支持 SVM, Logistic Regression, Random Forest
    """
    
    # 表3.4: 超参数网格搜索范围
    PARAM_GRIDS = {
        'SVM': {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'LR': {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']  # 支持L1和L2
        },
        'RF': {
            'n_estimators': [5, 10, 15, 20, 40, 60, 80, 100, 
                           200, 400, 600, 800, 1000],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                         14, 15, 20, 30, 40, 50, 60, None],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    def __init__(self):
        self.labeling_strategy = ClassificationLabelStrategy()
        self.feature_names = [  # 65个特征
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
    
    def process_single_scenario(self, 
                                input_folder: str,
                                solver: ColumnGenerationSolver = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单个场景: 加载数据+求解+特征提取+二分类标注
        
        Args:
            input_folder: input文件夹路径
            solver: 可选的求解器实例
            
        Returns:
            (特征矩阵, 二分类标签数组)
        """
        print(f"\n{'='*60}")
        print(f"处理场景: {input_folder}")
        print(f"{'='*60}")
        
        # 步骤1: 加载场景数据
        print("\n[步骤1] 加载场景数据...")
        loader = ScenarioLoader(input_folder)
        scenario_data = loader.load_all()
        
        schedule_df = scenario_data['schedule']
        parameters = scenario_data['parameters']
        maintenance_df = scenario_data['maintenance']
        
        # 步骤2: 初始化求解器
        if solver is None:
            solver = ColumnGenerationSolver()
        
        solver.load_from_dataframe(schedule_df, maintenance_df, parameters)
        
        # 步骤3: 列生成求解
        print("\n[步骤2] 列生成求解...")
        
        target_disrupted_ids = list(solver.target_disrupted_flights)[:1]
        non_target_disrupted_ids = list(solver.non_target_disrupted_flights)
        
        optimal_solution = solver.solve(
            target_disrupted_ids=target_disrupted_ids,
            non_target_disrupted_ids=non_target_disrupted_ids,
            use_ml_selection=False,
            max_iterations=50,
            scenario_path=input_folder
        )
        
        if not optimal_solution['converged']:
            print("  ⚠️ 未收敛,跳过此场景")
            return None, None
        
        print(f"  ✓ 求解完成")
        
        # 步骤4: 标注和特征提取
        all_features = []
        all_labels = []
        
        for disrupted_flight_id in target_disrupted_ids:
            print(f"\n[步骤3] 处理不正常航班 {disrupted_flight_id}")
            
            disrupted_flight = solver.flights[disrupted_flight_id]
            original_aircraft_id = disrupted_flight.original_aircraft_id
            
            candidate_aircraft = set(solver.aircraft.keys()) - {original_aircraft_id}
            candidate_aircraft_list = list(candidate_aircraft)
            
            # 二分类标注
            print(f"  [3.1] 二分类标注...")
            binary_labels = self.labeling_strategy.label_aircraft_binary(
                optimal_solution,
                disrupted_flight_id,
                original_aircraft_id,
                candidate_aircraft
            )
            
            # 特征提取
            print(f"  [3.2] 特征提取...")
            features_df = compute_features(
                schedule_df,
                disrupted_flight_id,
                original_aircraft_id,
                candidate_aircraft_list
            )
            
            if features_df.empty:
                print("    ⚠️ 特征提取失败")
                continue
            
            # 合并特征和标签
            X = features_df[self.feature_names].values
            y = [binary_labels[row['aircraft_id']] for _, row in features_df.iterrows()]
            
            all_features.append(X)
            all_labels.append(y)
            
            print(f"    ✓ 提取 {len(X)} 个样本")
        
        if not all_features:
            return None, None
        
        X_scenario = np.vstack(all_features)
        y_scenario = np.concatenate(all_labels)
        
        return X_scenario, y_scenario
    
    def train_model(self,
                   model_type: str,
                   scenario_folders: List[str],
                   output_model_path: str = None,
                   cv_folds: int = 5,
                   use_grid_search: bool = True) -> Dict:
        """
        训练单个分类模型
        
        Args:
            model_type: 'SVM', 'LR', 'RF'
            scenario_folders: 训练场景路径列表
            output_model_path: 模型保存路径
            cv_folds: 交叉验证折数
            use_grid_search: 是否使用网格搜索
            
        Returns:
            包含模型和评估结果的字典
        """
        print("\n" + "="*60)
        print(f"训练 {model_type} 分类模型")
        print("="*60)
        
        # 收集训练数据
        all_X = []
        all_y = []
        
        for i, input_folder in enumerate(scenario_folders, 1):
            print(f"\n处理场景 {i}/{len(scenario_folders)}")
            
            X, y = self.process_single_scenario(input_folder)
            
            if X is not None:
                all_X.append(X)
                all_y.append(y)
                print(f"  ✓ 场景{i}完成: {X.shape[0]}个样本")
            else:
                print(f"  ✗ 场景{i}跳过")
        
        if not all_X:
            print("\n❌ 没有有效的训练数据")
            return None
        
        # 合并数据
        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        
        print(f"\n{'='*60}")
        print("训练数据统计")
        print(f"{'='*60}")
        print(f"  总样本数: {X_train.shape[0]}")
        print(f"  特征维度: {X_train.shape[1]}")
        print(f"  正样本数: {np.sum(y_train == 1)}")
        print(f"  负样本数: {np.sum(y_train == 0)}")
        print(f"  正样本比例: {np.mean(y_train):.2%}")
        
        # 数据预处理
        X_train = np.nan_to_num(X_train, nan=0, posinf=999999, neginf=-999999)
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # 选择基础模型
        if model_type == 'SVM':
            base_model = SVC(random_state=42)
        elif model_type == 'LR':
            base_model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'RF':
            base_model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 网格搜索或直接训练
        if use_grid_search:
            print(f"\n{'='*60}")
            print(f"网格搜索最佳超参数 ({cv_folds}折交叉验证)")
            print(f"{'='*60}")
            
            param_grid = self.PARAM_GRIDS[model_type]
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='f1',  # 使用F1-Score作为评分标准
                n_jobs=-1,
                verbose=2
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print(f"\n最佳超参数: {best_params}")
            print(f"最佳交叉验证F1-Score: {best_score:.4f}")
        else:
            print(f"\n{'='*60}")
            print(f"使用默认参数训练")
            print(f"{'='*60}")
            
            best_model = base_model
            best_model.fit(X_train_scaled, y_train)
            best_params = best_model.get_params()
            best_score = None
        
        # 训练集评估
        y_pred_train = best_model.predict(X_train_scaled)
        
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_pred_train),
            'precision': precision_score(y_train, y_pred_train, zero_division=0),
            'recall': recall_score(y_train, y_pred_train, zero_division=0),
            'f1': f1_score(y_train, y_pred_train, zero_division=0)
        }
        
        print(f"\n{'='*60}")
        print("训练集评估结果 (表3.5)")
        print(f"{'='*60}")
        print(f"  Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall:    {train_metrics['recall']:.4f}")
        print(f"  F1-Score:  {train_metrics['f1']:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(y_train, y_pred_train)
        print(f"\n混淆矩阵:")
        print(f"  TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
        print(f"  FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")
        
        # 保存模型
        model_package = {
            'model': best_model,
            'scaler': scaler,
            'feature_names': self.feature_names,
            'model_type': model_type,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'cv_score': best_score
        }
        
        if output_model_path:
            with open(output_model_path, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"\n✅ 模型已保存至: {output_model_path}")
        
        return model_package
    
    def train_all_models(self,
                        scenario_folders: List[str],
                        output_dir: str = "models",
                        cv_folds: int = 5) -> Dict:
        """
        训练所有分类模型(SVM, LR, RF)
        
        Args:
            scenario_folders: 训练场景路径列表
            output_dir: 模型保存目录
            cv_folds: 交叉验证折数
            
        Returns:
            包含所有模型的字典
        """
        os.makedirs(output_dir, exist_ok=True)
        
        models = {}
        model_types = ['SVM', 'LR', 'RF']
        
        for model_type in model_types:
            output_path = os.path.join(output_dir, f"{model_type}_model.pkl")
            
            model_package = self.train_model(
                model_type=model_type,
                scenario_folders=scenario_folders,
                output_model_path=output_path,
                cv_folds=cv_folds,
                use_grid_search=True
            )
            
            if model_package:
                models[model_type] = model_package
        
        # 对比结果
        self._compare_models(models)
        
        return models
    
    def _compare_models(self, models: Dict):
        """对比不同模型的性能"""
        print(f"\n{'='*60}")
        print("模型性能对比 (表3.5)")
        print(f"{'='*60}")
        
        print(f"{'模型':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print("-" * 60)
        
        for model_type, package in models.items():
            metrics = package['train_metrics']
            print(f"{model_type:<10} "
                  f"{metrics['accuracy']:>10.4f} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} "
                  f"{metrics['f1']:>10.4f}")


def main():
    """主程序"""
    print("="*60)
    print("分类机器学习算法训练")
    print("="*60)
    
    # 场景路径
    scenario_folders = [
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819203/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819208/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819218/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819221/input',
        'D:/software/Engines/Rmops/rmops_rt/scenarios/AD/819229/input',
    ]
    
    # 过滤存在的文件夹
    existing_folders = [f for f in scenario_folders if os.path.exists(f)]
    
    if not existing_folders:
        print("❌ 未找到有效的场景文件夹！")
        return
    
    print(f"\n找到 {len(existing_folders)} 个有效场景")
    
    # 创建训练器
    trainer = ClassificationTrainer()
    
    # 训练所有模型
    models = trainer.train_all_models(
        existing_folders,
        output_dir="classification_models",
        cv_folds=5
    )
    
    if models:
        print("\n✅ 所有模型训练完成!")
        print("\n模型可用于:")
        print("  1. 二分类预测:飞机是否与恢复相关")
        print("  2. 子网络选择:过滤无关飞机")
        print("  3. 与排序模型对比分析")


if __name__ == "__main__":
    main()
