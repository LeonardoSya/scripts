import json
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt 
from pysal.explore import esda  
from pysal.lib import weights
from shapely.geometry import Point
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 数据读取与处理
with open('./data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data['数据'])
china = gpd.read_file('./2023年省级/2023年省级.shp', encoding='utf-8')

# 清理省份名称
china['省'] = china['省'].str.replace('省|市|自治区|特别行政区|维吾尔|壮族|回族', '', regex=True)
df['省份'] = df['省份'].str.replace('省|市|自治区|特别行政区|维吾尔|壮族|回族', '', regex=True)

# 合并数据
merged = china.merge(df, left_on='省', right_on='省份', how='left')

# 填充缺失的GDP数据为NaN
merged['GDP总量'] = merged['GDP总量'].fillna(float('nan'))
merged['增长率'] = merged['增长率'].fillna(float('nan'))

# 创建空间权重矩阵
w = weights.Queen.from_dataframe(merged)

# 计算Moran's I
moran_gdp = esda.Moran(merged['GDP总量'], w)
moran_growth = esda.Moran(merged['增长率'], w)
print(f"GDP总量的Moran's I: {moran_gdp.I:.4f}, p值: {moran_gdp.p_sim:.4f}")
print(f"增长率的Moran's I: {moran_growth.I:.4f}, p值: {moran_growth.p_sim:.4f}")

# 设置中文字体
font_path = '/System/Library/Fonts/PingFang.ttc'
font_prop = FontProperties(fname=font_path)
mpl.rcParams['font.family'] = font_prop.get_name()

# 创建图表
fig, ax = plt.subplots(figsize=(15, 10))

# 绘制GDP总量
merged.plot(column='GDP总量', cmap='YlOrRd', linewidth=0.8, edgecolor='0.8', ax=ax, legend=True, missing_kwds={'color': 'lightgrey'})
ax.set_title('2017年中国各省GDP总量和增长率', fontproperties=font_prop)
ax.axis('off')

# 添加文本标注显示增长率
for idx, row in merged.iterrows():
    if pd.notna(row['GDP总量']):
        ax.annotate(f"{row['增长率']:.1f}%", xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    xytext=(3, 3), textcoords="offset points", fontsize=8, fontproperties=font_prop)
    else:
        ax.annotate("无数据", xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                    xytext=(3, 3), textcoords="offset points", fontsize=8, fontproperties=font_prop)

# 保存和显示图表
plt.tight_layout()
plt.savefig('gdp_analysis_result.png', dpi=300, bbox_inches='tight')
plt.show()