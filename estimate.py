import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
# 输入的日期字符串
date_strings = Path("date_strings.txt").read_text().splitlines()

# 将字符串转换为datetime对象
dates = [datetime.datetime.strptime(d, "%Y.%m.%d") for d in date_strings]
dates.sort()  # 确保日期按时间顺序排列

# 计算相邻日期之间的间隔（天数）
intervals = []
for i in range(1, len(dates)):
    delta = dates[i] - dates[i-1]
    intervals.append(delta.days)

# 创建间隔序号（x轴）和间隔天数（y轴）
x = np.arange(1, len(intervals) + 1)  # 1, 2, 3, 4
y = np.array(intervals)

# 准备预测数据
future_points = 3  # 预测未来3个间隔
x_future = np.arange(len(intervals) + 1, len(intervals) + 1 + future_points)  # 5,6,7

# 线性回归预测
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
y_future = model.predict(x_future.reshape(-1, 1))

# 绘制原始间隔折线图
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', color='blue', linewidth=2, label='Actual Intervals')

# 绘制预测的虚线
plt.plot(x_future, y_future, 's--', color='red', linewidth=2, label='Predicted Intervals')

# 添加数据标签
for i, val in enumerate(y):
    plt.text(x[i], val + 1, f'{val} days', ha='center', va='bottom')
for i, val in enumerate(y_future):
    plt.text(x_future[i], val + 1, f'{int(round(val))} days', 
             ha='center', va='bottom', color='red')

# 添加标题和标签
plt.title('Date Intervals with Prediction', fontsize=14)
plt.xlabel('Interval Number', fontsize=12)
plt.ylabel('Days Between Dates', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')

# 设置x轴刻度
all_x = np.concatenate([x, x_future])
plt.xticks(all_x)

# 计算预测的未来日期
last_date = dates[-1]
future_dates = []
for i, days in enumerate(y_future):
    future_date = last_date + datetime.timedelta(days=int(round(days)))
    last_date = future_date  # 更新最后一个日期用于下一个预测
    future_dates.append(future_date.strftime("%Y.%m.%d"))

# 添加预测日期说明
plt.figtext(0.5, 0.01, 
            f"Predicted future dates: {', '.join(future_dates)}", 
            ha='center', fontsize=10, 
            bbox=dict(facecolor='yellow', alpha=0.3))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()