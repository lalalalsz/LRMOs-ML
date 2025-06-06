import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve

from matplotlib import rcParams
import warnings

#全局字体设置
enfont = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 13} #字体
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 13,
    'axes.unicode_minus': False  # 解决负号显示
})
warnings.filterwarnings("ignore", category=UserWarning)

# 机器学习模型
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV, cross_val_score #随机交叉验证
from sklearn.inspection import permutation_importance

#全局参数
RANDOM_STATE = 42       # 随机种子
TEST_SIZE = 0.2         # 验证集比例
FIG_SIZE = (8, 6)       # 图片尺寸
DPI = 300               # 图片分辨率
RESULT_PATH = 'F:/文件/本科/毕业设计/model_results/0.1C-XGB-10折.csv'
RESULT_PATH_1 = 'F:/文件/本科/毕业设计/model_results/RNC-参数.csv'

CV_FOLDS = 10                    # 交叉验证折数
PARAM_GRIDS = {}                # 各模型参数搜索空间
SELECTED_FOR_CV = []            # 选择要交叉验证的模型名
SELECTED_FOR_OPTIMIZE = []      # 选择要参数优化的模型名
SELECTED_FOR_IMPORTANCE = ['AdaBoost']    # 选择要特征分析的模型名
#'GBRT','XGBoost','Random Forest','AdaBoost','Decision Tree','KNN'
#数据集路径
def load_data(filepath, features,target):
    """
    加载数据集并进行预处理
    """
    data = pd.read_csv(filepath)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')] #去除Unnamed列
    
    X = data[features].apply(pd.to_numeric, errors='coerce')
    y = data[target].squeeze()
    
    
    # 数据标准化（可选，根据模型需求）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

#模型配置
def get_models():
    global PARAM_GRIDS

    base_models = {
        # 决策树
        'Decision Tree': DecisionTreeRegressor(
            max_depth=1,
            min_samples_leaf=9,#最小
            min_samples_split=9,#最
            random_state=RANDOM_STATE
        ),
        # 随机森林
        'Random Forest': RandomForestRegressor(
            n_estimators=300,
            max_depth=9,
            random_state=RANDOM_STATE
        ),
        # K近邻（新增标准化说明）
        'KNN': KNeighborsRegressor(
            n_neighbors=11,
            weights='distance',
            p=1
        ),
        # AdaBoost（新增学习率参数）
        'AdaBoost': AdaBoostRegressor(
            n_estimators=200,
            learning_rate=0.5,
            random_state=RANDOM_STATE
        ),
        # XGBoost（新增GPU支持选项）
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=4,
            random_state=RANDOM_STATE,
            learning_rate=0.05,
            subsample=0.8,
            #tree_method='hist'  # 可改为gpu_hist启用GPU加速
        ),
        # 梯度提升树
        'GBRT': GradientBoostingRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.01,
            random_state=RANDOM_STATE
        )
    }  
    PARAM_GRIDS = {
        'Decision Tree': {
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10], 
            'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200,300,400,500],
            'max_depth': [3, 5, 7, 9, 10,12,15,20],
        },
        'XGBoost': {
            'n_estimators': [100, 150,200,],
            'max_depth': [3, 4, 5, 6, 7, 8,9,10],
            'learning_rate': [0.01, 0.05,0.1,0.15,0.2,0.25,0.3],
            'subsample': [0.6,0.7,0.8, 1.0]
        },
        'AdaBoost': {
            'n_estimators': [50, 100,150,200],
            'learning_rate': [0.01,0.05,0.1,0.5,1.0]
        },
        'GBRT': {
            'n_estimators': [50, 100, 200,300,400,500],
            'max_depth': [1,2,3,4,5,6,7,8,9,10],
            'learning_rate': [0.01,0.05,0.1,0.15,0.2]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1: 曼哈顿距离, 2: 欧几里得距离
        }
    }
    return base_models

#可视化
def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """模型评估与可视化"""
    # 预测结果
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # 计算全局范围（修正关键错误）
    all_y = pd.concat([y_train, y_val])
    global_min = all_y.min()
    global_max = all_y.max()

    # 计算指标
    metrics = {
        'Train_R2': r2_score(y_train, y_train_pred),
        'Train_MAE': mean_absolute_error(y_train, y_train_pred),
        'Train_MSE': mean_squared_error(y_train, y_train_pred),
        'Val_R2': r2_score(y_val, y_val_pred),
        'Val_MAE': mean_absolute_error(y_val, y_val_pred),
        'Val_MSE': mean_squared_error(y_val, y_val_pred)
    }
    
    # 创建可视化
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # 验证集结果图
    plt.scatter(y_val, y_val_pred, alpha=0.5, c='#1f77b4')
    
    plt.plot([global_min, global_max], [global_min, global_max], '--', color='orange', linewidth=2)
    plt.xlabel('True Values', fontdict=enfont,fontsize=12)
    plt.ylabel('Predictions', fontdict=enfont,fontsize=12)
    plt.title(f'{model_name} Validation Performance\n'
              f"R2: {metrics['Val_R2']:.3f}  "
              f"MAE: {metrics['Val_MAE']:.3f}  "
              f"MSE: {metrics['Val_MSE']:.3f}")
    plt.grid(alpha=0.3)
    plt.savefig(f'result_images/{model_name}_validation.png', bbox_inches='tight')
    plt.close()
    
    # 训练集+验证集组合图
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    plt.scatter(y_train, y_train_pred, alpha=0.3, label='Train', color='#1f77b4')
    plt.scatter(y_val, y_val_pred, alpha=0.3, label='Validation', color='#ff7f0e')
    
    plt.plot([global_min, global_max], [global_min, global_max], '--', color='red', linewidth=2)
    plt.xlabel('True Values', fontdict=enfont,fontsize=12)
    plt.ylabel('Predictions', fontdict=enfont,fontsize=12)
    plt.legend()
    plt.title(f'{model_name} Combined Performance\n'
              f"Train R2: {metrics['Train_R2']:.3f}  "
              f"Val R2: {metrics['Val_R2']:.3f}")
    plt.grid(alpha=0.3)
    plt.savefig(f'result_images/{model_name}_combined.png', bbox_inches='tight')
    plt.close()
    
    return metrics

#自动参数优化
def param_optimization(model, param_grid):
    searcher = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring='r2',
        verbose=0
    )
    return searcher

#学习曲线
def plot_learning_curve(model, model_name, X_train, y_train):
    """绘制学习曲线"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=CV_FOLDS,
        scoring='MAE',
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # 计算统计量
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # 创建可视化
    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r",
             label="Training Score")
    plt.plot(train_sizes, val_mean, 'o-', color="g",
             label="Cross-Validation Score")
    
    # 图表装饰
    plt.xlabel("Training Examples", fontdict=enfont)
    plt.ylabel("R² Score", fontdict=enfont)
    plt.title(f"{model_name} Learning Curve\n"
              f"Final Train: {train_mean[-1]:.3f} ± {train_std[-1]:.3f} | "
              f"Final Val: {val_mean[-1]:.3f} ± {val_std[-1]:.3f}", fontdict=enfont)
    plt.legend(loc="best", prop=enfont)
    plt.grid(alpha=0.3)
    
    # 保存和关闭
    plt.savefig(f'result_images/{model_name}_learning_curve.png', bbox_inches='tight')
    plt.close()

#特征重要性可视化
def plot_feature_importance(model, feature_names, model_name):
    try:
        # 获取重要性数据
        importance = model.feature_importances_
    except AttributeError:
        print(f"⚠️ 警告：{model_name} 不支持特征重要性分析")
        return

    # 转换为numpy数组确保索引兼容性
    feature_names = np.array(feature_names)
    
    # 按重要性排序（从高到低）
    indices = np.argsort(importance)[::-1]
    sorted_importance = importance[indices]
    sorted_features = feature_names[indices]

    # 创建可视化
    plt.figure(figsize=(8, 6), dpi=300)  # 调整画布尺寸为横向布局
    
    # 绘制水平条形图
    plt.barh(
        y=np.arange(len(importance)),  # y轴位置
        width=sorted_importance,       # 条形长度
        align='center',                # 居中对齐
        color='#1f77b4',               # 保持颜色一致
        height=0.8                     # 调整条带高度
    )
    
    # 设置纵坐标（特征名称）
    plt.yticks(
        ticks=np.arange(len(importance)),
        labels=sorted_features,
        fontsize=10,
        fontproperties=enfont
    )
    
    # 设置横坐标
    plt.xlabel("重要性得分", fontsize=12, labelpad=10, fontname='SimSun')
    plt.ylabel("特征名称", fontsize=12, labelpad=10,fontname='SimSun')
    #plt.title(f"{model_name} - 特征重要性排序", fontsize=14, pad=20,fontname='SimSun')
    
    for i, v in enumerate(sorted_importance):
        if v > 0:
            plt.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')

    
    # 优化布局
    plt.gca().invert_yaxis()  # 重要特征显示在顶部
    plt.grid(axis='x', alpha=0.3)  # 添加横向网格线
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f"{model_name}_feature_importance.png", bbox_inches='tight')
    plt.close()

#主程序
if __name__ == "__main__":
    # 数据加载
    DATA_PATH = 'F:/文件/本科/毕业设计/数据/数据库-填充.csv'
    
    features = [
        #'0.1C',
        '1C',
        #'R','N','C',
        #'充电电压','放电电压',
        'dope','cover',
        'ICE',
        'I/I',
        'charge_Li', 'charge_O', 'charge_F',
        'charge_Na', 
        'charge_Mg', 
        'charge_Al', 
        'charge_S', 'charge_K', 'charge_Mn',
        'charge_Fe', 'charge_Co', 'charge_Ni',
        'charge_Zr', 'charge_W'
    ]
    target = '0.1C'
    
    X_train, X_val, y_train, y_val = load_data(DATA_PATH, features, target)
    
    # 获取模型配置
    models = get_models()
    results = []
    best_params_list = []
    # 遍历训练模型
    for name, model in models.items():
        
        # 训练模型
        model.fit(X_train, y_train)
        
        #学习曲线
        #plot_learning_curve(model, name, X_train, y_train)

        # 参数自动优化
        if name in SELECTED_FOR_OPTIMIZE:
            searcher = param_optimization(model, PARAM_GRIDS[name])
            searcher.fit(X_train, y_train)
            model = searcher.best_estimator_
            best_params_list.append({'Model': name, 'Parameters': str(model)})
            print(f"{name} 最佳参数：{searcher.best_params_}")
        
        # 交叉验证评估
        if name in SELECTED_FOR_CV:
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=CV_FOLDS, scoring='r2'
            )
            print(f"{name} 交叉验证R2：{np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")
        
        # 特征重要性分析
        if name in SELECTED_FOR_IMPORTANCE:
            plot_feature_importance(model, features, name)
        
        # 评估和可视化
        metrics = evaluate_model(model, X_train, X_val, y_train, y_val, name)
        results.append((name, metrics))
        
        # 打印进度
        print(f'{name} 训练完成')
        print(f"验证集 R2: {metrics['Val_R2']:.3f}")
    
    # 输出结果概览
    print("\n模型性能对比：")
    for name, metrics in results:
        print(f"{name:12} | Val R2: {metrics['Val_R2']:.3f} | Val MAE: {metrics['Val_MAE']:.3f}")

    # 结果对比（新增排序功能）
    print("\n\n模型性能对比（按验证集R2排序）：")
    sorted_results = sorted(results, key=lambda x: x[1]['Val_R2'], reverse=True)
    for idx, (name, metrics) in enumerate(sorted_results, 1):
        print(f"{idx:2}. {name:15} | R2: {metrics['Val_R2']:.3f} | "
              f"MAE: {metrics['Val_MAE']:.3f} | "
              f"MSE: {metrics['Val_MSE']:.3f}")
    
    # 结果输出到CSV
    # 将结果转换为DataFrame
    results_df = pd.DataFrame([
        {
            'Model': name,
            **metrics
        } for name, metrics in sorted_results  # 使用排序后的结果
    ])
    
    # 调整列顺序
    column_order = ['Model',
                    'Train_R2', 'Val_R2',
                    'Train_MAE', 'Val_MAE',
                    'Train_MSE', 'Val_MSE']
    results_df = results_df[column_order]
    
    # 格式美化：保留三位小数
    float_cols = results_df.select_dtypes(include=['float']).columns
    results_df[float_cols] = results_df[float_cols].round(3)
    
    # 保存CSV文件
    #results_df.to_csv(RESULT_PATH, index=False, encoding='utf-8-sig')
    #if best_params_list:
    #    pd.DataFrame(best_params_list).to_csv(RESULT_PATH_1, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至：{RESULT_PATH}")