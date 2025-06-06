import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import r2_score
from tqdm import tqdm
import numpy as np
# ==== 设置路径 ====
base_dir = os.getcwd()
#csv_path = os.path.join(base_dir, "data", "data.csv")
csv_path = 'F:/文件/本科/毕业设计/数据/数据库-填充.csv'
#output_dir = os.path.join(base_dir, "results")
output_dir = 'F:/文件/本科/毕业设计/SHAP1/'
os.makedirs(output_dir, exist_ok=True)

# ==== 1. 加载数据 ====
print("📥 正在加载数据...")
df = pd.read_csv(csv_path)
feature_cols = [
        '0.1C',
        '1C',
        #'R','N','C',
        #'充电电压','放电电压',
        'dope','cover',
        #'ICE',
        'I/I',
        'charge_Li', 'charge_O', 'charge_F',
        'charge_Na', 
        'charge_Mg', 
        'charge_Al', 
        'charge_S', 'charge_K', 'charge_Mn',
        'charge_Fe', 'charge_Co', 'charge_Ni',
        'charge_Zr', 'charge_W']
target_col = 'ICE'
X = df[feature_cols]
y = df[target_col]

# ==== 2. 划分训练集 / 测试集 ====
print("✂️ 划分训练测试集...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 3. 模型训练 ====
print("🏋️‍♂️ 正在训练 XGBoost 模型...")
'''
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.2,
    subsample=0.8,
    #colsample_bytree=0.8,
    random_state=42
)
'''
model = AdaBoostRegressor(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42
        )
model.fit(X_train, y_train)

# ==== 4. 模型预测 ====
print("🔮 预测测试集...")
y_pred = model.predict(X_test)

# ==== 5. SHAP 分析 ====
print("🧠 计算 SHAP 值...")
#explainer = shap.TreeExplainer(model)
#shap_values = explainer.shap_values(X)

#background = shap.sample(X_train, 50)  # 通过采样减少计算量
explainer = shap.KernelExplainer(model.predict, X_train, n_jobs=-1)  # 使用训练集作为背景数据
shap_values = explainer.shap_values(X)  # 建议在测试集上计算以提升速度



# ==== 6. 可视化部分 ====
## 6.1 实际 vs 预测
plt.figure(figsize=(6, 6))
# 计算 R² 和 RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Observed NPP")
plt.ylabel("Predicted NPP")
plt.title("Observed vs Predicted")
# 显示 R² 和 RMSE
plt.text(0.05, 0.95, f"R² = {r2:.4f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
plt.text(0.05, 0.90, f"RMSE = {rmse:.4f}", fontsize=12, ha='left', va='top', transform=plt.gca().transAxes)
# 格式调整
plt.grid(False)
plt.tight_layout()
# 保存图像
plt.savefig(os.path.join(output_dir, "scatter_observed_vs_predicted_with_R2_RMSE.png"))
plt.close()

## 6.2 SHAP 蜂巢图（violin）
shap.summary_plot(shap_values, X, plot_type="violin", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_beeswarm.png"))
plt.close()

## 6.3 SHAP summary 图（bar）
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"))
plt.close()

## 6.4 SHAP Force 图（第一个样本）
shap.initjs()
force_plot = shap.plots.force(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_force_plot.png"))
plt.close()

## 6.5 SHAP Waterfall 图（第一个样本）
shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                      base_values=explainer.expected_value,
                                      data=X.iloc[0].values,
                                      feature_names=feature_cols),
                     show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_waterfall_plot.png"))
plt.close()

## 6.6 SHAP 多项式拟合依赖图（每个变量）
print("📈 生成多项式拟合依赖图...")
for i, feature in enumerate(tqdm(feature_cols, desc="多项式拟合")):
    shap_val = shap_values[:, i]
    feature_val = X[feature].values

    plt.figure(figsize=(6, 4))
    sns.regplot(
        x=feature_val,
        y=shap_val,
        order=2,
        scatter_kws={'alpha': 0.2, 's': 10},
        line_kws={'color': 'red', 'linewidth': 2}
    )
    plt.xlabel(feature)
    plt.ylabel(f"SHAP value for {feature}")
    plt.title(f"Dependence Plot with Polynomial Fit: {feature}")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_dependence_poly_{feature}.png"))
    plt.close()

## 6.7 SHAP 默认交互依赖图
print("🔀 生成 SHAP 交互依赖图...")
for feature in tqdm(feature_cols, desc="交互依赖图"):
    shap.dependence_plot(feature, shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shap_dependence_interact_{feature}.png"))
    plt.close()

# ==== 7. 输出模型性能 ====
r2 = r2_score(y_test, y_pred)
print(f"✅ 模型训练完成，R² = {r2:.4f}")
print(f"📂 所有图表已保存到：{output_dir}")
