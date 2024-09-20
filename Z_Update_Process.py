########################### 导入需要的库 #######################
import pandas as pd
import numpy as np
import datetime
import openpyxl
import glob
import os
from datetime import datetime
from openpyxl import load_workbook
import re
from itertools import groupby
import akshare as ak
import tushare as ts
import warnings  
import new_data_source as ds
import json
import pymysql
# from mycache import DailyCache
# import update_mysql as um
from decimal import Decimal
from functools import lru_cache
from sqlalchemy import create_engine
from DjData import ths
warnings.filterwarnings(action='ignore', category=pd.errors.SettingWithCopyWarning)



################## 数据处理函数 ####################
## 判断日期在第几期
def check_period(target_date, start_list, end_list):  
    # 将字符串日期转换为datetime对象，方便比较  
    target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")  
    
    # 遍历所有日期对，找到目标日期所在的周期  
    for start, end in zip(start_list, end_list):  
        start_date_obj = datetime.strptime(start, "%Y-%m-%d")  
        end_date_obj = datetime.strptime(end, "%Y-%m-%d")  
        
        # 检查目标日期是否在范围内  
        if start_date_obj <= target_date_obj <= end_date_obj:  
            return start_list.index(start) + 1 

        else:  
            continue 
             


# 修改第一、第二期的数据
def transfer(frame):  
    # 第一期  
    try:  
        # 找出在指定时间范围内且'project'为'IF-日内多'或'IF-日内空'的行  
        mask_first_period_special = (frame['委托时间'] >= '2024-05-13') & (frame['委托时间'] <= '2024-05-18') & frame['project'].isin(['IF-日内多', 'IF-日内空'])  
        # 更新这些行的'平仓盈亏'列  
        frame.loc[mask_first_period_special, '平仓盈亏'] = frame.loc[mask_first_period_special, '平仓盈亏'] / 2 * 5  
        # 找出在指定时间范围内但'project'不是'IF-日内多'或'IF-日内空'的行  
        mask_first_period_other = (frame['委托时间'] >= '2024-05-13') & (frame['委托时间'] <= '2024-05-18') & ~frame['project'].isin(['IF-日内多', 'IF-日内空'])  
        # 更新这些行的'平仓盈亏'列  
        frame.loc[mask_first_period_other, '平仓盈亏'] = frame.loc[mask_first_period_other, '平仓盈亏'] / 2  
    except Exception as e:  
        print(f'第一期数据处理时出错: {e}')  
  
    
    # 第二期  
    try:  
        # 找出在指定时间范围内且'project'为'IF-日内多'或'IF-日内空'的行  
        mask_first_period_special = (frame['委托时间'] >= '2024-05-20') & (frame['委托时间'] <= '2024-05-25') & frame['project'].isin(['IF-日内多', 'IF-日内空'])  
        # 更新这些行的'平仓盈亏'列  
        frame.loc[mask_first_period_special, '平仓盈亏'] = frame.loc[mask_first_period_special, '平仓盈亏'] / 2 * 5  
    except Exception as e:  
        print(f'第二期数据处理时出错: {e}')  
    return frame  # 返回处理后的frame

# 转换日期格式
def transfer_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')  
    formatted_date = date_obj.strftime('%Y%m%d')  
    return formatted_date



def filter_by_time_2(df, start_date, end_date):
    
    if 'date' not in df.columns:  
        raise ValueError("DataFrame does not contain a column named 'date'.")  
      
    try:  
        df['date'] = pd.to_datetime(df['date'])  
    except ValueError as e:  
        raise ValueError("Failed to convert 'date' column to datetime: " + str(e))  
    
    try:  
        start_date = pd.to_datetime(start_date)  
        end_date = pd.to_datetime(end_date)  
        # 将end_date设置为这一天的午夜之后的时间（即下一天的开始），然后减去一个时间差，得到这一天的午夜  
        end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  
    except ValueError as e:  
        raise ValueError("Failed to parse start_date or end_date: " + str(e))  
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]


def filter_by_time(df, start_date, end_date):
    
    if '委托时间' not in df.columns:  
        raise ValueError("DataFrame does not contain a column named '委托时间'.")  
      
    try:  
        df['委托时间'] = pd.to_datetime(df['委托时间'])  
    except ValueError as e:  
        raise ValueError("Failed to convert '委托时间' column to datetime: " + str(e))  
    
    try:  
        start_date = pd.to_datetime(start_date)  
        end_date = pd.to_datetime(end_date)  
        # 将end_date设置为这一天的午夜之后的时间（即下一天的开始），然后减去一个时间差，得到这一天的午夜  
        end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  
    except ValueError as e:  
        raise ValueError("Failed to parse start_date or end_date: " + str(e))  
    return df[(df['委托时间'] >= start_date) & (df['委托时间'] <= end_date)]


# 定义一个函数来生成新期数的元组列表  
def generate_new_periods(start, end, strategies=['IM-日内多', 'IM-日内空', 'IM-综合']):  
    new_tuples = []  
    for period in range(start):  
        for strategy in strategies:  
            new_tuples.append((f'第{period}期数据', strategy))  
    return new_tuples  


from datetime import datetime, timedelta, date  
def enddate_plus_one(end_date):
    # 将字符串转换为日期对象  
    date_obj = datetime.strptime(end_date, "%Y-%m-%d").date()  
    # 从日期对象中减去一天  
    new_date_obj = date_obj + timedelta(days=1)  
    # 如果需要，将新的日期对象转换回字符串  
    new_date_str = new_date_obj.strftime("%Y-%m-%d")  
    return new_date_str


# 定义一个函数来检查 DataFrame 是否为空  
def is_empty_df(df):  
    return df.empty  

# 判断start_date和end_date之间有多少期
def period_num(start_date, end_date):
    num = 0
    num = end_list.index(end_date) - start_list.index(start_date) + 1
    return num

# 根据期数、project类型设置账户初始金额
def cash_initial(project, s_date, e_date):
    cash = 0
    num = period_num(start_date=s_date, end_date=e_date)
    # 设置project对应的乘数
    if project == 'IF-综合' or project == 'IM-综合':
        project_num = 2
    else:
        project_num = 1
    cash = 500000 * num * project_num  
    return cash

def group_people(total_people):  
    # 计算每组的基础人数（向下取整）  
    base_people = total_people // 5  
      
    # 如果余数不为0，说明需要调整组的人数  
    remainder = total_people % 5  
      
    # 初始化分组人数  
    groups = [0] * 5  
      
    # 对于第1、2、4、5组，如果余数不为0，则每组加1，否则直接等于基础人数  
    for i in [0, 1, 3, 4]:  
        groups[i] = base_people + (1 if remainder > 0 else 0)  
      
    # 根据余数调整第3组的人数  
    # 如果余数不为0，第3组需要减去已经加到其他四组的额外人数  
    # 如果余数为0，则第3组就是基础人数  
    groups[2] = total_people - sum(groups[:2]) - sum(groups[3:])  
      
    # 确保第3组的人数不大于其他四组  
    # 如果第3组人数大于其他组，且总人数允许，则重新分配  
    if groups[2] > groups[0] and total_people > 4 * (groups[0] + 1):  
        # 尝试从第1、2、4、5组中各减1人给第3组，但不超过它们的原始分配  
        for i in [0, 1, 3, 4]:  
            if groups[i] > base_people:  
                groups[2] += 1  
                groups[i] -= 1  
    return groups


# 定义一个函数计算基准收益
# def market(project, start_date, end_date):
#     start_date = transfer_date(start_date)
#     end_date = transfer_date(end_date)
#     temp_index = ak.stock_zh_index_daily_em(symbol='sz399300', start_date=start_date, end_date=end_date)
#     if project=='IF-日内空' or project=='IF-日内多' or project=='IF-综合':
#         df_index = pro.fut_daily(ts_code="IFL.CFX", start_date=start_date, end_date=end_date).sort_values(by='trade_date', ascending=True)
#     else:
#         df_index = pro.fut_daily(ts_code="IML.CFX", start_date=start_date, end_date=end_date).sort_values(by='trade_date', ascending=True)
#     df_index['return'] = df_index['close'] / df_index['pre_close'] - 1
#     df_index['Cum'] = (1+df_index['return']).cumprod()
#     df_index['return'] = df_index['return'] * 100
#     try:
#         df_index.index = temp_index['date']
#     except:
#         pass
#     return df_index


def market(project, start_date, end_date):  
    temp_index = ak.stock_zh_index_daily_em(symbol='sz399300', start_date=start_date, end_date=end_date)  
      
    # 根据项目类型选择期货合约  
    if project in ('IF-日内空', 'IF-日内多', 'IF-综合'):  
        symbol = 'IF'  
    else:  
        symbol = 'IM'  
      
    df_index = ths.history_future(symbol=symbol, start_time='2024-05-10', end_time='2030-01-01', period='1d', df=True)[['eob', 'close', 'open', 'high', 'low']]  
      
    # 重命名列  
    df_index.rename(columns={'eob': 'date'}, inplace=True)  
      
    # 确保 'date' 列是 datetime 类型  
    df_index['date'] = pd.to_datetime(df_index['date'])  
      
    # 转换日期格式并创建新列  
    df_index['trade_date'] = df_index['date'].dt.strftime("%Y%m%d")  
      
    # 计算前一天收盘价  
    df_index['pre_close'] = df_index['close'].shift(1)  
      
    # 根据 start_date 和 end_date 过滤数据  
    df_index = filter_by_time_2(df_index, start_date=start_date, end_date=end_date)  
    # 计算收益率  
    df_index['return'] = df_index['close'] / df_index['pre_close'] - 1  
    
      
    # 计算累计收益  
    df_index['Cum'] = (1 + df_index['return']).cumprod()  
    df_index['return'] = df_index['return'] * 100  # 将收益率转换为百分比  
      
    # 尝试将 index 设置为 temp_index 的日期（如果 temp_index 存在且包含 'date' 列）  
    if 'date' in temp_index.columns:  
        df_index.index = pd.to_datetime(temp_index['date'])  
      
    return df_index 



# 定义归一化函数  
def normalize(rank_series):  
    min_rank = rank_series.min()  
    max_rank = rank_series.max()  
    return (rank_series - min_rank) / (max_rank - min_rank)


class DateEncoder(json.JSONEncoder):
    """
    功能：格式化日期
    """
    def default(self, o):
        if isinstance(o, datetime):
            return o.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(o, date):
            return o.strftime("%Y-%m-%d")
        elif isinstance(o, Decimal):
            return float(o)
        else:
            return json.JSONEncoder.default(self, o)


class NanEncoder:
    def _nan_to_none(self, data):
        """
        将数据中的 NaN 值转换为 None。
        :param data: 要处理的数据，可以是列表、字典。
        """
        if isinstance(data, dict):
            return {k: self._nan_to_none(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._nan_to_none(x) for x in data]
        elif pd.isna(data):
            return None
        return data



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
pro = ts.pro_api('818670fa68bc204c217143cdb75efeae1986031841ff8ca2c6a855bd')
ts.set_token('818670fa68bc204c217143cdb75efeae1986031841ff8ca2c6a855bd')

# @DailyCache
def data_get():
    # 在日内数据库中获取数据
    database_uri = 'mysql+pymysql://root:djct003@192.168.10.210/dayin'
    engine = create_engine(database_uri)
    conn = pymysql.connect(
            host='192.168.10.210',  # 数据库主机名,3号虚拟机
            port=3306,               # 数据库端口号，默认为3306
            user='root',             # 数据库用户名
            passwd='djct003',         # 数据库密码
            db='dayin',               # 数据库名称
            charset='utf8'           # 字符编码
        )
    # 创建游标对象
    cur = conn.cursor()
    # 查询数据库中的所有表名
    cur.execute("SHOW TABLES")
    # 获取所有表名
    table_names = [row[0] for row in cur.fetchall()]
    
    # 创建一个空字典来存储表名和DataFrame  
    tables_dict = {}  
    
    # 遍历表名，读取每个表的数据到DataFrame，并存入字典  
    for table_name in table_names:  
        try:  
            # 读取表数据到DataFrame  
            df = pd.read_sql_table(table_name, engine)  
            # 将表名和DataFrame存入字典  
            tables_dict[table_name] = df  
        except Exception as e:  
            print(f"Error reading table {table_name}: {e}")  
    return tables_dict


tacitic = ds.TacticRace('dzkj002','djct666666')

# 生成开始和结束日期列表  
df = tacitic.query_project()
filtered_df = df[df['projectName'].str.contains('日内', na=False)]  
project_Id_list = filtered_df['projectId'].tolist()

# 所有人
all_name = tacitic.query_participant(project_Id_list[-1])['userName'].tolist()
start_list = []
end_list = []
for project_id in project_Id_list:
    topic_df = tacitic.query_topic(project_id)
    start_list.append(topic_df['start_date'].iloc[0])
    end_list.append(topic_df['end_date'].iloc[0])
end_list[4] = '2024-06-14'

# 可选类型
sheet_list = ['IF-日内多', 'IF-日内空',
                'IM-日内多', 'IM-日内空']

IF_list = ['IF-日内空', 'IF-日内多']
IM_list = ['IM-日内空', 'IM-日内多']



total = data_get()

# 创建一个新的字典和列表来存储结果，以避免在迭代时修改原始数据结构  
new_total = {}  
new_all_name = []  

# 迭代 total 字典  
for key, value in total.items():  
    if is_empty_df(value):  
        pass  

    else:  
        new_value = transfer(value)  
        new_total[key] = new_value  
        new_all_name.append(key)  

total = new_total  


class Tactic:
    """策略数据

        参数
        ----------
        start_date :
            开始时间
            str
            
        end_date :
            结束时间
            str
            
        project :
            策略选择
            str

            ['IF-日内多', 'IF-日内空', 'IM-日内多', 'IM-日内空', 'IF-综合','IM-综合']

        all_name :
            所有人
            list

        name : 
            个人
            str

    """
    def __init__(self, start_date='2024-07-22', end_date='2024-07-26', project='IM-综合', all_name=new_all_name, name=None) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.end_date_plus = enddate_plus_one(end_date)
        self.project = project
        self.all_name = all_name
        self.name = name


    def person_metric(self, df_market):
        """
        个人指标计算
        ----------
        Returns:
            dataframe: 个人拥有所有指标的和每日的总金额信息
        """
        data = total[self.name]
        
        if self.project=='IF-综合':
            column = ['IF-日内多', 'IF-日内空']

        elif self.project=='IM-综合':
            column = ['IM-日内多', 'IM-日内空']  

        else:
            column = [self.project]

        # 尝试将'委托时间'列转换为datetime类型  
        data['date'] = pd.to_datetime(data['委托时间'], errors='coerce')  # errors='coerce'会将无法转换的值设置为NaT（Not a Time）  

        mask_first_period_other = (data['date'] >= self.start_date) & (data['date'] <= self.end_date_plus) & data['project'].isin(column)
        data = data[mask_first_period_other]
        data = data.drop_duplicates()
        

        # 现在，'委托时间'列应该包含datetime对象或NaT。我们可以安全地应用strftime。  
        data['date_str'] = data['date'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else None)  
        
        ### 通过看所选日期中包含多少个开始日期来决定初始比赛金额
        cash_list = []
        date_list = []
        cash = cash_initial(self.project, self.start_date, self.end_date)
        grouped = data.groupby(data['date_str'])

        
        for date, group in data.groupby('date_str'):  
            date_list.append(date)
            net_daily = group['平仓盈亏'].sum() - group['手续费'].sum()
            
            cash += net_daily
            cash_list.append(cash)

        df_cash = pd.DataFrame({
            'cash': cash_list
        }, index=date_list)
        
        

        # 每日收益率
        df_cash['return'] = (df_cash['cash'] / df_cash['cash'].shift(1) - 1) * 100
        # 假设 cash_initial 是初始的 cash 值  
        try:
            df_cash['return'].iloc[0] = (df_cash['cash'].iloc[0] / cash_initial(self.project, self.start_date, self.end_date) - 1) * 100
        except:
            pass

        # 累计收益率
        df_cash['Cum'] = (1+df_cash['return'] * 0.01).cumprod()

        # 计算最大回撤
        df_cash["previous_max"] = df_cash["cash"].cummax(axis=0)
        df_cash['draw_downs'] = df_cash['cash'] / df_cash['previous_max'] - 1
        max_drawdown = df_cash['draw_downs'].min() * 100
                
        # 总盈利
        profit = data[data['平仓盈亏']>=0]['平仓盈亏'].sum()

        # 总亏损
        loss = data[data['平仓盈亏']<0]['平仓盈亏'].sum()

        # 手续费
        commission = data['手续费'].sum()

        # 净利润
        net = profit + loss - commission

        # 当期收益率
        return_ratio = (net / cash_initial(self.project, self.start_date, self.end_date))

        # 盈亏比
        pl = abs(profit / loss)

        

        # 交易次数（一买一卖等于一次）
        trade_times = data[data['成交数量'] == 1]['成交数量'].sum()

        # 盈利比例
        if trade_times != 0:
            profit_ratio = len(data[data['平仓盈亏']>0]) / trade_times * 100
        else:
            profit_ratio = 0

        # 亏损比例
        loss_ratio = len(data[data['平仓盈亏']<0]) / trade_times * 100
        # 每组平均收益
        avg_return = (profit + loss - commission) / trade_times
        # 每笔平均盈利
        avg_profit = profit / len(data[data['平仓盈亏']>=0])
        # 每笔平均亏损
        avg_loss = loss / len(data[data['平仓盈亏']<0])
        # 平均盈利 / 平均亏损
        avg_ratio = abs(avg_profit / avg_loss)
        # 计算天数
        day = data.groupby('date')
        # 平均开平组数（天）
        avg_times = trade_times / len(day)
        # 平均持仓周期(分钟)
        data['diff'] = data['date'] - data['date'].shift(1)
        time = data.loc[data.index % 2 == 1, 'diff'].sum()
        total_minutes = int(time.total_seconds() / 60)
        avg_hold = total_minutes / trade_times
        # 持仓时间比例
        if len(day) != 0:
            hold_rate = total_minutes / (len(day) * 24 * 60) * 100
        else:
            hold_rate = 0
        
        # 抵御风险能力
        ## 收益波动率
        volatility = np.std(df_cash['return']) / np.mean(df_cash['return'])

        ## 夏普比率
        # df_cash['index_Cum'] = df_market['Cum']
        # sharp = (df_cash['Cum'].iloc[-1] - df_cash['index_Cum'].iloc[-1]) / volatility

        ## 计算每日涨幅占振幅的比例
        df_market['amplitude'] = (df_market['high'] - df_market['low']) / df_market['open']
        df_market['increase_in_amplitude'] = abs(df_market['return']) / df_market['amplitude'] 
        df_cash['increase_in_amplitude'] = df_market['increase_in_amplitude']
        df_cash['index_return'] = df_market['return']

        # 趋势日收益率
        condition = ((df_cash['index_return'] > 0.01) | (df_cash['index_return'] < -0.01)) & (df_cash['increase_in_amplitude'] > 0.6)
        df_cash['whether_trend'] = 0
        df_cash.loc[condition, 'whether_trend'] = 1 
        trend_ratio = df_cash.loc[df_cash['whether_trend'] == 1]['return'].mean()

        # 震荡日收益率
        df_amplitude = df_cash[~condition]['return']
        amplitude_ratio = df_amplitude.mean() 

        # 数据格式处理
        df = pd.DataFrame({
            '收益率': return_ratio,
            '净利润': round(net, 2),
            '总盈利': round(profit, 2),
            '总亏损': round(loss, 2),
            '盈亏比': round(pl, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '交易次数': round(trade_times, 2),
            '盈利比例': round(profit_ratio, 2),
            '亏损比例': round(loss_ratio, 2),
            '每笔平均收益': round(avg_return, 2),
            '每笔平均盈利': round(avg_profit, 2),
            '每笔平均亏损': round(avg_loss, 2),
            '平均盈亏比': round(avg_ratio, 2),
            '平均开平组数/天': round(avg_times, 2),
            '平均持仓周期/分钟': round(avg_hold, 2),
            '持仓时间比例': round(hold_rate, 2),
            '波动率': round(volatility, 2),
            # '夏普比率': round(sharp, 2),
            # '趋势日收益率': round(trend_ratio, 2),
            # '震荡日收益率': round(amplitude_ratio, 2),
            '类型': self.project
        }, index=[0])    
        return df, df_cash
    

    def new_metric(self, name, project):
        """
        个人指标计算
        ----------
        Returns:
            dataframe: 个人比赛开始以来的所有数据计算
        """
        data = total[name]
        
        if project=='IF-综合':
            column = ['IF-日内多', 'IF-日内空']

        elif project=='IM-综合':
            column = ['IM-日内多', 'IM-日内空']  

        else:
            column = [project]

        # 尝试将'委托时间'列转换为datetime类型  
        data['date'] = pd.to_datetime(data['委托时间'], errors='coerce')  # errors='coerce'会将无法转换的值设置为NaT（Not a Time）  

        mask_first_period_other = (data['date'] >= self.start_date) & (data['date'] <= self.end_date_plus) & data['project'].isin(column)
        data = data[mask_first_period_other]
        

        # 现在，'委托时间'列应该包含datetime对象或NaT。我们可以安全地应用strftime。  
        data['date_str'] = data['date'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else None)  
        
        ### 通过看所选日期中包含多少个开始日期来决定初始比赛金额
        cash_list = []
        date_list = []
        cash = cash_initial(project, self.start_date, self.end_date)
        grouped = data.groupby(data['date_str'])

        
        for date, group in data.groupby('date_str'):  
            date_list.append(date)
            net_daily = group['平仓盈亏'].sum() - group['手续费'].sum()
            
            cash += net_daily
            cash_list.append(cash)

        df_cash = pd.DataFrame({
            'cash': cash_list
        }, index=date_list)
        
        
        df_market = market(project, self.start_date, self.end_date)
        # 每日收益率
        df_cash['return'] = (df_cash['cash'] / df_cash['cash'].shift(1) - 1) * 100
        df_cash['index_return'] = df_market['return']
        # 假设 cash_initial 是初始的 cash 值  
        try:
            df_cash['return'].iloc[0] = (df_cash['cash'].iloc[0] / cash_initial(project, self.start_date, self.end_date) - 1) * 100
        except:
            pass

        # 获取趋势日  
        df_day = self.whether_trend()
        df_trend = df_day.loc[df_day['whether_trend']==1]
        df_amplitude = df_day.loc[df_day['whether_trend']==0]
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        trend_list = formatted_dates_str.to_list()   # 综合趋势日列表
    
        # 趋势多
        df_long = df_trend.loc[df_trend['index_return'] > 0]
        long_list = df_long.index.to_list()   # 趋势多列表
        

        # 趋势空 
        df_short = df_trend.loc[df_trend['index_return'] < 0]
        short_list = df_short.index.to_list()   # 趋势空列表

        # 震荡
        amplitude_list = df_amplitude.index.to_list()   # 震荡列表

        # 趋势日收益率
        trend_data = df_cash.loc[df_cash.index.isin(trend_list)]
        attack_ratio = trend_data['return'].mean()

        # 震荡日收益率
        amplitude_data = df_cash.loc[df_cash.index.isin(amplitude_list)]
        defence_ratio = amplitude_data['return'].mean()
        return attack_ratio, defence_ratio


    @lru_cache    
    def all_member_data(self):
        """
        所有数据
         ----------

        Returns: tuple  
            dataframe:
                总表

            dict:
                key: 成员名, 
                value: dataframe:
                    index: 日期, 
                    column: 各项指标
        """
        # 初始化
        all_member_dict = {}
        all_member_cash = {}

        df_market = market(self.project, self.start_date, self.end_date)
        for name in self.all_name:
            still = Tactic(start_date=self.start_date, end_date=self.end_date, project=self.project, all_name=self.all_name, name=name)
            try:
                data, df_cash = still.person_metric(df_market)
                all_member_dict[name] = data
                all_member_cash[name] = df_cash
            except:
                pass
                # print(f'{name}数据获取失败')
        
        # 初始化
        key_list = []
        value_list = []
        df = pd.DataFrame()

        for key, value in all_member_dict.items():
            key_list.append(key)
            value_list.append(value)
        
        # 构建总体dataframe
        df = pd.concat(value_list, axis=0)
        df.index = key_list
        # df['趋势日收益率'].fillna(0, inplace=True)  
        df = df.dropna()

        # 收益率排名
        df['return_rank'] = df['收益率'].rank(method='min', ascending=False) 
        return df, all_member_cash
    
    def person_return(self):
            """
            累计收益（折线图）
            ----------
            Returns:
                dict
                    key: 人名
                    value: 累计收益的dataframe
                
                    dataframe
                        index: 日期
                        value: 个人的收益
            """
            person_cum_dict = {}
            df_market = market(self.project, self.start_date, self.end_date)
            for person in self.all_name:
                try:
                    people = Tactic(start_date=self.start_date, end_date=self.end_date, project=self.project, name=person)
                    temp, person_cum = people.person_metric(df_market)
                    person_cum = person_cum[['return']]
                    person_cum.columns = ['收益率']
                    person_cum_dict[person] = person_cum
                except:
                    print(f'{person}数据获取失败')
            return person_cum_dict
        

    def return_ratio_grouped(self, group_num):
        """
        每期总体平均收益率（折线图）及大于基准收益率人数的百分比(直方图)
         ----------
        Returns:
            dataframe:
                index: 第i期, i=1,2,3...
                column: 总体平均收益率、大于基准收益率人数百分比
        """
        demo = Tactic(start_date=self.start_date, end_date=self.end_date, project=self.project, all_name=self.all_name, name=None)
        df_all, df_all_cash = demo.all_member_data() 
  
        # 使用qcut进行分组，返回的是Categorical类型  
        df_all['group'] = pd.qcut(df_all['return_rank'], q=group_num, labels=False, duplicates='drop')  
  
         
        labels = [f'group_{i}' for i in range(group_num)]  
        df_all['group_label'] = pd.qcut(df_all['return_rank'], q=group_num, labels=labels, duplicates='drop')
        
        grouped = df_all.groupby(df_all['group_label'])
        dict_list = []
        for group_name, group in grouped:
            member_dict = {}
            member = group.index.tolist()
            for man in member:
                member_dict[man] = df_all_cash[man]
            
            # 将每个人的每日资产放入列表中
            dict_list.append(member_dict)

        daily_averages = {}
        for i, group in enumerate(dict_list):
            daily_avg_df = pd.DataFrame()

            for name, df in group.items():
                if daily_avg_df.empty:
                    daily_avg_df['return'] = df['return']
                else:
                    daily_avg_df['return'] += df['return']

            daily_avg_df['return'] /= len(group)

            daily_averages[f'Group{i+1}'] = daily_avg_df

        merge_df = pd.concat(daily_averages.values(), axis=1)
        columns_list = list(daily_averages.keys())
        
        merge_df.columns = columns_list
        merge_df= pd.DataFrame(merge_df.values * 0.01, columns=merge_df.columns, index=merge_df.index) 
        for i in range(1, group_num+1):
            merge_df[f'Group{i}'] = (1+merge_df[f'Group{i}']).cumprod()
        merge_df = merge_df.applymap(lambda x: f'{x:.3}') 
        return merge_df
    

    def IM_curve(self):
        
        demo = Tactic(start_date=self.start_date, end_date=self.end_date, project=self.project, all_name=self.all_name, name=None)
        df_all, df_all_cash = demo.all_member_data() 
  

        dict_list = []
        return_list = {}
        for key, value in df_all_cash.items():
            dict_list.append(value)
        
        merge_df = pd.concat(dict_list, axis=0)
        grouped = merge_df.groupby(merge_df.index)
        df_market = market(self.project, self.start_date, self.end_date)
        for date, group in grouped:
            return_list[date] = round(group['return'].sum() / len(group), 3)
        df = pd.DataFrame.from_dict(return_list, orient='index').T
        df.index = ['每日平均收益率']
        df = df.T  
        df['基准收益率'] = round(df_market['return'] * 100, 2)
        return df


    def all_comparison(self):
        """
        每期总体平均收益率（折线图）及大于基准收益率人数的百分比(直方图)
        返回一个dataframe
        index为第i期, i=1,2,3...
        column为总体平均收益率、大于基准收益率人数百分比
        """
        avg_list = []
        exceed_num_list = []
        period_num_list = []
        index_return_list = []
        df_all_comparison = pd.DataFrame()
        for i in range(0, len(start_list)-1):
            temp = Tactic(start_date=start_list[i], end_date=end_list[i], project=self.project, all_name=self.all_name, name=None)
            df_all, df_all_cash = temp.all_member_data()
            avg_return = df_all['收益率'].mean()
            avg_list.append(avg_return)
            
            # 该期基准收益率
            df_index = market(self.project, start_list[i], end_list[i])
            index_return = (df_index.iloc[-1]['close'] / df_index.iloc[0]['pre_close'] - 1)
            # index_return = (df_index.iloc[-1]['close'] / df_index.iloc[0]['open'] - 1)
            index_return_list.append(index_return)

            # 超过基准收益率的人数百分比
            if self.project == 'IM-日内多' or self.project == 'IF-日内多':
                exceed_num = len(df_all[df_all['收益率'] > index_return])
            else:
                exceed_num = len(df_all[df_all['收益率'] > abs(index_return)])
            
            try:
                exceed_num = exceed_num / len(df_all)
                exceed_num_list.append(exceed_num)
                period_num_list.append(f'第{i+1}期')
            except:
                exceed_num = 0
                exceed_num_list.append(exceed_num)
                period_num_list.append(f'第{i+1}期')

        df_all_comparison['总体平均收益率'] = avg_list
        df_all_comparison['大于基准收益率人数百分比'] = exceed_num_list
        df_all_comparison['基准收益率'] = index_return_list
        df_all_comparison.index = period_num_list
        df_all_comparison['总体平均收益率'] = round(df_all_comparison['总体平均收益率'] * 100, 2)
        df_all_comparison['基准收益率'] = round(df_all_comparison['基准收益率'] * 100, 2)
        df_all_comparison['大于基准收益率人数百分比'] = df_all_comparison['大于基准收益率人数百分比'].apply(lambda x: f'{x:.0%}')
        df_all_comparison = df_all_comparison[['总体平均收益率', '基准收益率', '大于基准收益率人数百分比']]
        return df_all_comparison 
    

    
    def all_single_comparison(self):
        """
        每期总体平均收益率（折线图）及大于基准收益率人数的百分比(直方图)
         ----------
        Returns:
            dataframe:
                index: 第i期, i=1,2,3...
                column: 总体平均收益率、大于基准收益率人数百分比
        """
        
        data = {  
            '收益平均值': [None],  # 或者使用pd.NA（pandas 1.0.0及以上版本）或float('nan')  
            '上涨人数百分比': [None],  
            '下跌人数百分比': [None],  
            '上涨平均值': [None],  
            '下跌平均值': [None],
            '基准收益率': [None]
        }
        df_all_comparison = pd.DataFrame(data)  
        
        temp = Tactic(start_date=self.start_date, end_date=self.end_date, project=self.project, all_name=self.all_name, name=None)
        df_all, df_all_cash = temp.all_member_data()
        df_all_comparison['收益平均值'].iloc[0] = (df_all['收益率'].mean() * 100).round(2)
        
        # 涨跌平均值
        df_all_comparison['上涨平均值'].iloc[0] = (df_all[df_all['收益率']>0]['收益率'].mean() * 100).round(2)
        df_all_comparison['下跌平均值'].iloc[0] = (df_all[df_all['收益率']<=0]['收益率'].mean() * 100).round(2)
        df_market = market(self.project, self.start_date, self.end_date)
        a = df_market['close'].iloc[-1] / df_market['open'].iloc[0] - 1
        df_all_comparison['基准收益率'].iloc[0] = round(a * 100, 2)

        # 涨跌人数统计
        df_all_comparison['上涨人数百分比'].iloc[0] = round(len(df_all[df_all['收益率']>0]) / len(df_all) * 100, 2)
        df_all_comparison['下跌人数百分比'].iloc[0] = round(len(df_all[df_all['收益率']<=0]) / len(df_all) * 100, 2)
        
        df_all_comparison = df_all_comparison[['收益平均值', '上涨平均值', '下跌平均值', '基准收益率', '上涨人数百分比', '下跌人数百分比']]
        df_all_comparison.index = [f'第{start_list.index(self.start_date)+1}期']
        return df_all_comparison 
    
    

    def comparison_radar(self, period_num_1, period_num_2):
        period_list = [period_num_1, period_num_2]
        df_1 = pd.DataFrame()

        for period_num in period_list:
            list_1 = []
            demo_1 = Tactic(start_date=start_list[period_num-1], end_date=end_list[period_num-1], project=self.project, all_name=self.all_name, name=self.name)

            df_total_radar_1, df_cash_radar_1 = demo_1.all_member_data()
            a = df_total_radar_1['收益率'].mean()
            b = df_total_radar_1['risk'].mean()
            c = df_total_radar_1['趋势日收益率'].mean()
            d = df_total_radar_1['震荡日收益率'].mean()
            list_1 = [a, b, c, d]
            df_1[f'period_{period_num}'] = list_1
        df_1.index = ['整体收益率', '防御能力', '趋势日收益率', '震荡日收益率']
        return df_1



    ################## 个人模式 ##################
    def person_radar(self):
        """
        个人综合表现（雷达图）
         ----------
        Returns:
            dict:
                key为人名, 
                value为column是上述指标的dataframe
                column值: attack, defense, compre, profit_ratio, win_ratio, stability
        """
        person = {}

        df_all, df_all_cash = self.all_member_data()

        for name in df_all.index:
            person[name] = df_all.loc[name][['return_rank', 'risk', 'trend_rank', 'amplitude_rank']]
        return person

    
    def person_horizontal_comparison(self):
        """
        个人横向对比
         ----------
        Returns:
            dict:
                key: 人名
                value: 个人四种策略排名的dataframe

                    dataframe的column:
                        'IF-5日多', 'IF-5日空', 'IM-5日多', 'IM-5日空'
        """
        project_list = ['IF-日内多', 'IF-日内空', 'IM-日内多', 'IM-日内空']
        horizontal_dict = {}
        horizontal_list = []
        for project in project_list:
            demo = Tactic(start_date=self.start_date, end_date=self.end_date, project=project, all_name=self.all_name, name=None)
            df_all, df_all_cash = demo.all_member_data()
            horizontal_list.append(df_all)

        df_total = pd.concat(horizontal_list)
        grouped = df_total.groupby(df_total.index)
        for name, group in grouped:
            try:
                group.index = project_list
                horizontal_dict[name] = group['return_rank']

            except:
                print('该期缺乏数据')
        return horizontal_dict
    

    


    def person_single_rank_tail(self, num):
        """
        个人一期单个策略排名后三名
         ----------
        Returns:
            dict:
                key: 策略名称
                value: dataframe

                dataframe:
                    index: 人名
                    value: 收益率
                
        """
        # project_list = ['IF-日内多', 'IF-日内空', 'IM-日内多', 'IM-日内空']
        project_list = ['IM-日内多', 'IM-日内空', 'IM-综合']
        horizontal_dict = {}
        horizontal_list = []
        for project in project_list:
            demo = Tactic(start_date=self.start_date, end_date=self.end_date, project=project, all_name=self.all_name, name=None)
            df_all, df_all_cash = demo.all_member_data()
            df_all = df_all.sort_values(by = 'return_rank')
            df_all = df_all.tail(num)
            market_index = market(project, self.start_date, self.end_date)
            market_index['index_return'] = market_index['close'].iloc[-1] / market_index['open'].iloc[0] - 1
            df_all['index_return'] = market_index['index_return'].iloc[0]
            df_all['收益率'] = df_all['收益率'].apply(lambda x: f'{x:.2%}')
            df_all['基准收益率'] = df_all['index_return'].apply(lambda x: f'{x:.2%}')
            df_all['排名'] = df_all['return_rank']
            # new_row = pd.Series([None, df_all['基准收益率'].iloc[0]], index=['排名', '收益率'])  
            # new_row_df = pd.DataFrame([new_row]) # 转置使索引成为列名，第一维（行）索引是我们想要的  
            # new_row_df.index = ['基准收益率']  # 显式设置索引  
            
            # # 使用concat合并DataFrame  
            # df_all = pd.concat([df_all, new_row_df], axis=0)  # axis=0表示沿着行的方向合并  
            horizontal_dict[project] = df_all[['排名', '收益率']]
        return horizontal_dict

    def person_single_rank_head(self, num):
        """
        
        Returns:
            个人一期单个策略排名前三名

            dict:
                key: 策略名称
                value: dataframe
                index: 人名
                value: 收益率
        """
        # project_list = ['IF-日内多', 'IF-日内空', 'IM-日内多', 'IM-日内空']
        project_list = ['IM-日内多', 'IM-日内空', 'IM-综合']
        horizontal_dict = {}
        horizontal_list = []
        for project in project_list:
            demo = Tactic(start_date=self.start_date, end_date=self.end_date, project=project, all_name=self.all_name, name=None)
            df_all, df_all_cash = demo.all_member_data()
            df_all = df_all.sort_values(by = 'return_rank')
            df_all = df_all.head(num)
            market_index = market(project, self.start_date, self.end_date)
            market_index['index_return'] = market_index['close'].iloc[-1] / market_index['open'].iloc[0] - 1
            df_all['index_return'] = market_index['index_return'].iloc[0]
            df_all['收益率'] = df_all['收益率'].apply(lambda x: f'{x:.2%}')
            df_all['基准收益率'] = df_all['index_return'].apply(lambda x: f'{x:.2%}')
            df_all['排名'] = df_all['return_rank']
            # new_row = pd.Series([None, df_all['基准收益率'].iloc[0]], index=['排名', '收益率'])  
            # new_row_df = pd.DataFrame([new_row]) # 转置使索引成为列名，第一维（行）索引是我们想要的  
            # new_row_df.index = ['基准收益率']  # 显式设置索引  
            
            # 使用concat合并DataFrame  
            # df_all = pd.concat([df_all, new_row_df], axis=0)  # axis=0表示沿着行的方向合并  
            horizontal_dict[project] = df_all[['排名', '收益率']]
        return horizontal_dict

    def person_cum_return(self):
        """
        累计收益（折线图）
         ----------
        Returns:
            dict
                key: 人名
                value: 累计收益的dataframe
            
            dataframe
                index: 日期
                value: 个人的累计收益
        """
        person_cum_dict = {}
        df_market = market(self.project, self.start_date, self.end_date)
        for person in self.all_name:
            people = Tactic(start_date=self.start_date, end_date=self.end_date, project=self.project, name=person)
            temp, person_cum = people.person_metric(df_market)
            person_cum = person_cum[['Cum']]
            person_cum.columns = ['累计收益率']
            person_cum['name'] = person 
            person_cum_dict[person] = person_cum
        return person_cum_dict
    

    


    def person_return_rank(self):
        """
        单个策略的每日排名（折线图）
         ----------
        Returns: 
            dict
                key: 人名
                value: 每日排名的dataframe
        
            dataframe:
                index: 日期
                column: 收益率, name, rank
        """
        person_cum_dict = {}
        df_market = market(self.project, self.start_date, self.end_date)
        for person in self.all_name:
            people = Tactic(start_date=self.start_date, end_date=self.end_date, project=self.project, name=person)
            temp, person_cum = people.person_metric(df_market)
            person_cum = person_cum[['return', 'whether_trend']]
            person_cum.columns = ['收益率', '是否为趋势日']
            person_cum['name'] = person 
            person_cum_dict[person] = person_cum

        # 将字典转换为DataFrame  
        value_list = []
        for key, value in person_cum_dict.items():
            value_list.append(value)
            
        df = pd.concat(value_list, axis=0)
        df = df.dropna()
        df.index.name = '日期'  # 明确索引名  

        grouped_date = df.groupby(df.index)
        group_list = []
        for date, group in grouped_date:
            group['rank'] = group['收益率'].rank(method='min', ascending=False)
            group_list.append(group)

        df = pd.concat(group_list, axis=0)

        grouped_name = df.groupby(df['name'])
        dict_of_dfs = {}  
  
        # 遍历按'name'分组的GroupBy对象  
        for name, group in df.groupby('name'):  
            dict_of_dfs[name] = group  
        return dict_of_dfs



    def whether_trend(self):
        """
        Returns:
            dataframe
                index: 日期
                column: return(指数收益率), whether_trend(是否为趋势日) 
        """
        df_market = market(self.project, self.start_date, self.end_date)
        df_market['index_return'] = df_market['close'] - df_market['pre_close']
        # df_market['index_return'] = df_market['close'] - df_market['open']
        df_market['amplitude'] = df_market['high'] - df_market['low'] 
        condition = (abs(df_market['index_return']) > 100) & (abs(df_market['amplitude']) > 100)
        df_market['whether_trend'] = 0
        df_market.loc[condition, 'whether_trend'] = 1 
        
        # df_market = df_market[['return', 'whether_trend']]
        return df_market

    ################## 对比模式 ##################
    """
    各项指标对比（雷达图）
    在个人综合表现, 即person_radar()返回的所有人的指标字典处取两人数据


    累计收益曲线（折线图）
    在个人模式的累计收益曲线处person_cum_return返回的所有人的累计收益曲线字典处取两人数据
    """


    ################## 总榜数据 #################
    def total_data(self):
        IM_list = ['IM-日内多', 'IM-日内空', 'IM-综合']
        all_dict = {}
        for project in IM_list:
            aa_list = []
            for i in range(0, 8):
                demo_1 = Tactic(start_date=start_list[i], end_date=end_list[i], project=project, all_name=all_name, name=None)
                aa, bb = demo_1.all_member_data()
                aa = aa['收益率'].round(4) * 100
                aa.name = '第{}期'.format(i+1)  
                aa_list.append(aa)
                result_df = pd.concat(aa_list, axis=1, join='outer') 
                result_df['总收益'] = result_df.sum(axis=1)
                result_df['参加期数'] = result_df.drop(columns='总收益').count(axis=1)
                result_df['每期平均收益'] = (result_df['总收益'] / result_df['参加期数']).round(2)
                col_data = result_df.pop('每期平均收益')
                result_df.insert(0, '每期平均收益', col_data)
                col_data = result_df.pop('参加期数')
                result_df.insert(0, '参加期数', col_data)
                col_data = result_df.pop('总收益')
                result_df.insert(0, '总收益', col_data)
            all_dict[project] = result_df

        df_im_intra_long = all_dict['IM-日内多']
        df_im_intra_short =all_dict['IM-日内空']
        df_im_comprehensive = all_dict['IM-综合']
        group_df = pd.concat([df_im_intra_long, df_im_intra_short, df_im_comprehensive], axis=1).T
        grouped = group_df.groupby(group_df.index)

        # 总榜数据
        group_list = []
        for name, group in grouped:
            group.index = ['IM-日内多', 'IM-日内空', 'IM-综合']
            group_list.append(group)


        columns_original = pd.MultiIndex.from_tuples([  
            ('参加期数', 'IM-日内多'), ('参加期数', 'IM-日内空'), ('参加期数', 'IM-综合'),  
            ('总收益', 'IM-日内多'), ('总收益', 'IM-日内空'), ('总收益', 'IM-综合'),  
            ('每期平均收益率', 'IM-日内多'), ('每期平均收益率', 'IM-日内空'), ('每期平均收益率', 'IM-综合'),  
        ])  

        for i in range(0, len(start_list)):
            new_columns = generate_new_periods(i+1)
            # 将新生成的元组列表添加到原始多级索引中  
            columns_expanded = pd.MultiIndex.from_tuples(columns_original.tolist() + new_columns) 

        final_df = pd.concat(group_list, axis=0)
        final_df.index = columns_expanded
        final_df = final_df.T
        # final_df.to_excel('总榜数据(5日).xlsx')
        return final_df
    


    ############### 新增 ##############
    def single_rank(self, comprehensive_project:str):
        """单期综合前三和后三名

        Params:
            comprehensive_project:
                'IM-综合' or 'IF-综合'

        Return:
            dataframe:
                index: 人名
                column: 排名、收益率
        """

        df_comprehensive_1 = self.person_single_rank_head(3)[comprehensive_project]  
        df_comprehensive_2 = self.person_single_rank_tail(3)[comprehensive_project]  
          
        # 合并头部和尾部数据  
        df_total = pd.concat([df_comprehensive_1, df_comprehensive_2], axis=0)  
        return df_total
    


    
    


    def attack_high_low(self, comprehensive_project:str, date_i:int): 
        """最好与最坏进攻收益

        params:
            comprehensive_project: 'IF-综合', 'IM-综合'
            date_i: 第几期

        Return:
            dataframe:
                index: 姓名
                column: 排名、收益率
        """
        if comprehensive_project == 'IF-综合':
            comprehensive_list = ['IF-日内多', 'IF-日内空']
        else:
            comprehensive_list = ['IM-日内多', 'IM-日内空']
        
        demo_1 = Tactic(start_date=start_list[date_i], end_date=end_list[date_i], project=comprehensive_project, all_name=all_name, name=None)  
        
        # 获取最优进攻排名  
        df_comprehensive_1 = demo_1.person_single_rank_head(1)[comprehensive_list[0]]  
        df_comprehensive_2 = demo_1.person_single_rank_head(1)[comprehensive_list[1]]  
        
        # 获取最差进攻排名  
        df_comprehensive_3 = demo_1.person_single_rank_tail(1)[comprehensive_list[0]]   
        df_comprehensive_4 = demo_1.person_single_rank_tail(1)[comprehensive_list[1]]  
        
        # 合并最优和最差数据  
        df_top = pd.concat([df_comprehensive_1, df_comprehensive_2], axis=0)  
        df_bottom = pd.concat([df_comprehensive_3, df_comprehensive_4], axis=0)  
        
        # 分别取最优和最差的顶部记录  
        df_top_best = df_top.sort_values(by='收益率', ascending=False).iloc[0:1]  # 假设只取前1名  
        df_bottom_worst = df_bottom.sort_values(by='收益率', ascending=True).iloc[0:1]  # 假设只取最差的1名  
        
        # 合并最优和最差记录  
        df_final = pd.concat([df_top_best, df_bottom_worst], axis=0)  
        return df_final
    


    def attack_defend(self, project:str):
        """总体进攻与防守收益
        
        Params: 
            project: 'IM-综合' or 'IF-综合'

        Return:
            dataframe:
                index: 第几期
                column: IM/IF-日内多、IM/IF-日内空、IM/IF-进攻、IM/IF-防守
        """
        if project == 'IM-综合':
            project_list = ['IM-日内多', 'IM-日内空']
        else:
            project_list = ['IF-日内多', 'IF-日内空']
        demo_1 = Tactic(start_date=start_list[1], end_date=end_list[1], project=project_list[0], all_name=all_name, name=None)
        df1 = demo_1.all_comparison()

        demo_1 = Tactic(start_date=start_list[1], end_date=end_list[1], project=project_list[1], all_name=all_name, name=None)
        df2 = demo_1.all_comparison()

        df_merge_total = pd.DataFrame()
        df_merge_total[project_list[0]] = df1['总体平均收益率']
        df_merge_total[project_list[1]] = df2['总体平均收益率']
        df_merge_total.index = df1.index
        df_merge_total['进攻'] = df_merge_total.apply(lambda row: row[project_list[0]] if row[project_list[0]] > row[project_list[1]] else row[project_list[1]], axis=1)
        df_merge_total['防守'] = df_merge_total.apply(lambda row: row[project_list[1]] if row[project_list[1]] < row[project_list[0]] else row[project_list[0]], axis=1)
        return df_merge_total
    


    def attack_static(self, comprehensive_project):
        """
        根据综合项目计算静态攻击策略的结果。

        Args:
            self (object): 类实例对象。
            comprehensive_project (str): 综合项目名称，可选值为'IF-综合'或'IM-综合'。

        Returns:
            pd.DataFrame: 合并后的DataFrame,包含综合收益占比优秀的人数统计和反向收益占比的人数统计。

        """
        if comprehensive_project == 'IF-综合':
            comprehensive_list = ['IF-日内多', 'IF-日内空']
        else:
            comprehensive_list = ['IM-日内多', 'IM-日内空']
        # IM振幅
        amplitude_dict_max = {}
        amplitude_dict_min = {}
        for i in range(0, len(start_list)-1):
            demo1 = Tactic(start_date=start_list[i], end_date=end_list[i], project=comprehensive_project, all_name=all_name, name=None) 
            df_all_member, df_all_member_cash = demo1.all_member_data() 
            df_initial = market(comprehensive_project, start_list[i], end_list[i])
            df_initial['close_yesterday'] = df_initial['close'].shift(1)
            df_initial['close_yesterday'].iloc[0] = df_initial['open'].iloc[0]
            y = (df_initial['high'] - df_initial['low']).mean() / df_initial['close_yesterday'].mean() * 100

            df_attack = df_all_member[['收益率']] * 100
            amplitude_dict_max[f'第{i+1}期'] = len(df_attack[df_attack['收益率'] / (2 * y) >= 0.8])
            amplitude_dict_min[f'第{i+1}期'] = len(df_attack[df_attack['收益率'] / (2 * y) <= -0.8])


        df_1 = pd.DataFrame.from_dict(amplitude_dict_max, orient='index').T
        df_1.index = ['综合收益占比优秀的人数统计']
        df_2 = pd.DataFrame.from_dict(amplitude_dict_min, orient='index').T
        df_2.index = ['反向收益占比的人数统计']
        df_merge = pd.concat([df_1, df_2], axis=0)
        df_merge = df_merge.T
        return df_merge
    


    





#################################################################################### 新野 ##################################################################################
####################### 总体模式 ###########################
    def daily_return(self, project:str):
            """返回策略每日的平均收益率

            Params:
                project:
                    选择策略

            Return:
                dataframe:
                    index: 日期
                    column: 每日平均收益率、基准收益率、每日累计收益率
            """
            daily_list = []
            for i in range(0, len(start_list)-1):
                demo_1 = Tactic(project=project, start_date=start_list[i], end_date=end_list[i], all_name=self.all_name, name=None)
                df_daily = demo_1.IM_curve()
                daily_list.append(df_daily)
            df_project = pd.concat(daily_list, axis=0)
            df_project['每日累计收益率'] = df_project['每日平均收益率'].cumsum()
            df_project['每日累计基准收益率'] = df_project['基准收益率'].cumsum()
            return df_project

    def return_grouped_period(self, group_num=5): 
        group_1_list = []
        group_2_list = []
        group_3_list = []
        group_4_list = []
        group_5_list = []
        period_list = []
        df_group = pd.DataFrame()
        for i in range(0, len(start_list)-1):
            temp = Tactic(start_date=start_list[i], end_date=end_list[i], project=self.project, all_name=self.all_name, name=None)
            df_all, df_all_cash = temp.all_member_data()
            df_all.sort_values(by=['收益率'], ascending=True, inplace=True)
            num_list = group_people(len(df_all))
            group_1 = df_all.iloc[:num_list[0]]
            group_2 = df_all.iloc[num_list[0]:num_list[0]+num_list[1]]
            group_3 = df_all.iloc[num_list[0]+num_list[1]:num_list[0]+num_list[1]+num_list[2]]
            group_4 = df_all.iloc[num_list[0]+num_list[1]+num_list[2]:num_list[0]+num_list[1]+num_list[2]+num_list[3]]
            group_5 = df_all.iloc[num_list[0]+num_list[1]+num_list[2]+num_list[3]:]
            
            

            # 计算每组平均收益率  
            group_1_return = group_1['收益率'].mean()  
            group_2_return = group_2['收益率'].mean()
            group_3_return = group_3['收益率'].mean()
            group_4_return = group_4['收益率'].mean()
            group_5_return = group_5['收益率'].mean()

            group_1_list.append(group_1_return)
            group_2_list.append(group_2_return)
            group_3_list.append(group_3_return)
            group_4_list.append(group_4_return)
            group_5_list.append(group_5_return)
            period_list.append(f'第{i+1}期')
        
        df_group = pd.DataFrame(
            {
                'group_1': group_1_list,
                'group_2': group_2_list,
                'group_3': group_3_list,
                'group_4': group_4_list,
                'group_5': group_5_list
            }, index=period_list
        )
        return df_group


    # 按部门排名
    def return_department(self):
        """
        部门平均收益率
         ----------
        Returns:
            dataframe:
                index: 部门名称
                column: 部门平均收益率      
        """

        department_dict = {
            '数据部': ['谢佳钊', '肖炎辉', '范创', '高昌盛', '徐晓朋', '庄斯杰'],
            '投资部': ['张炳坤', '刘亚洲', '伍水明', '杨宝圣', '肖健', '谭泽松'],
            '软件部': ['郭文彬', '李陇杰', '孔德枞', '罗嘉俊', '屈中翔', '阮云泉', '陶松', '阳俊'],
            '开发部': ['林秋宇', '欧佰康', '孙彪', '游凯文', '张晓光', '周思捷'],
            '总经办-技术部': ['章魏康', '钟远金', '唐硕', '陈广', '赵明昊', '刘俊浩', '何博轩', '陈楷锐', '郭婉婷', '郭总', '黄永朗', 
                        '黄梓聪', '赖晓梅', '罗威', '庞优华', '王畅', '张湘子', '张紫荷', '郑妮斯', '黄慧仪', ]
        }
        data_list = []
        invest_list = []
        software_list = []
        develop_list = []
        tech_list = []
        period_list = []
        df_department = pd.DataFrame()
        for i in range(0, len(start_list)-1):
            temp = Tactic(start_date=start_list[i], end_date=end_list[i], project=self.project, all_name=self.all_name, name=None)
            df_all, df_all_cash = temp.all_member_data()
            df_data = df_all.loc[df_all.index.isin(department_dict['数据部'])]['收益率'].mean()
            df_invest = df_all.loc[df_all.index.isin(department_dict['投资部'])]['收益率'].mean()
            df_software = df_all.loc[df_all.index.isin(department_dict['软件部'])]['收益率'].mean()
            df_develop = df_all.loc[df_all.index.isin(department_dict['开发部'])]['收益率'].mean()
            df_tech = df_all.loc[df_all.index.isin(department_dict['总经办-技术部'])]['收益率'].mean()
            data_list.append(df_data)
            invest_list.append(df_invest)
            software_list.append(df_software)
            develop_list.append(df_develop)
            tech_list.append(df_tech)
            period_list.append(f'第{i+1}期')
        df_department = pd.DataFrame({
            '数据部': data_list,
            '投资部': invest_list,
            '软件部': software_list,
            '开发部': develop_list,
            '总经办-技术部': tech_list
        }, index=period_list)
        return df_department
    


    ## 收益分布情况
    def trend_for_day(self):  
        """  
        趋势排行榜  
    
        Returns:  
            dataframe:  
                index: 选手名称  
                column: 每个趋势日的进攻收益  
        """  
        long_dict = {}
        short_dict = {}
        
        for i in range(0, len(start_list)-1):
            demo_long = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-日内多', all_name=self.all_name, name=None)  
            long_temp = demo_long.person_return() 
            for name, df in long_temp.items():
                if name not in long_dict:
                    long_dict[name] = df
                else:
                    long_dict[name] = pd.concat([long_dict[name], df])
                

            demo_short = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-日内空', all_name=self.all_name, name=None)  
            short_temp = demo_short.person_return() 
            for name, df in short_temp.items():
                if name not in short_dict:
                    short_dict[name] = df
                else:
                    short_dict[name] = pd.concat([short_dict[name], df])

            
        
        demo_temp = Tactic(start_date=start_list[0], end_date=end_list[-1], project='IM-日内多', all_name=self.all_name, name=None)
        # 获取趋势日  
        df_trend = demo_temp.whether_trend()
        df_trend = df_trend.loc[df_trend['whether_trend']==1]
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        trend_list = formatted_dates_str.to_list()   # 趋势日列表 
    
        # 初始化结果DataFrame  
        result_df = pd.DataFrame(index=long_dict.keys(), columns=trend_list)  
    
        # 遍历每个选手的DataFrame  
        for name, df in long_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_df.at[name, trend_date] = 0  
        df_long_dict = {col: result_df[col] for col in result_df.columns}  
        for key, value in result_df.items():
            value = value.dropna()
            df_long_dict[key] = value

        # 初始化结果DataFrame  
        result_short_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
        # 遍历每个选手的DataFrame  
        for name, df in short_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_short_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_short_df.at[name, trend_date] = 0  
        df_short_dict = {col: result_short_df[col] for col in result_short_df.columns}  
        for key, value in result_short_df.items():
            value = value.dropna()
            df_short_dict[key] = value       
        
        result_dict = {}
        for key in df_short_dict.keys():  
            # 使用concat横向拼接DataFrame，并设置新的列名  
            df_combined = pd.concat([df_long_dict[key], df_short_dict[key]], axis=1)  
            # 重命名列  
            if df_trend.loc[key]['index_return'] > 0:
                df_combined.columns = ['进攻', '防守']  
            else:
                df_combined.columns = ['防守', '进攻']  
            df_combined = df_combined[['进攻']]
            df_index = pro.fut_daily(ts_code='IM.CFX', start_date=transfer_date(key), end_date=transfer_date(key))
            # 当日期货涨幅
            change = (df_index['close'] / df_index['pre_close'] - 1).iloc[0] * 100
            swing = (df_index['high'].iloc[0] - df_index['low'].iloc[0]) / df_index['pre_close'].iloc[0] * 100
            # change = (df_index['close'] / df_index['open'] - 1).iloc[0] * 100
            # swing = (df_index['high'].iloc[0] - df_index['low'].iloc[0]) / df_index['open'].iloc[0] * 100
            
            # 参赛人数
            num = len(df_combined)
            top = len(df_combined[df_combined['进攻'] > swing*2*0.8])
            middle = len(df_combined[(df_combined['进攻'] < swing*2*0.8) & (df_combined['进攻'] > swing*2*0.6)])
            down = num - top - middle

            # 占比
            top_rate = top/num
            middle_rate = middle/num
            down_rate = down/num
            df_comp = pd.DataFrame({
                '>80%': [top],
                '60%-80%': [middle],
                '<60%': [down]
            }, index=['人数'])

            # 将结果存储回结果字典  
            df_comp.reset_index(inplace=True, drop=False)
            df_comp.rename(columns={'index': 'Num'}, inplace=True)
            result_dict[key] = df_comp.to_dict(orient='records')
        # 转换后的数据结构  
        converted_data = []  
        
        # 遍历原始数据  
        for date, records in result_dict.items():  
            # 因为每个日期对应的记录列表只有一个元素，所以直接取第一个元素  
            record = records[0]  
            # 从record中移除'Time'键（如果不需要的话），因为它在所有记录中都是相同的  
            del record['Num']  
            # 添加日期键到record中  
            record['Date'] = date  
            # 将更新后的record添加到结果列表中  
            converted_data.append(record)  
        return converted_data
    


#################################### 个人 #######################################

    def person_single_rank(self):
        """
        个人一期单个策略排名
         ----------
        Returns:
            dict:
                key: 策略名称
                value: dataframe

                dataframe:
                    index: 人名
                    value: 收益率
                
        """
        # project_list = ['IF-日内多', 'IF-日内空', 'IM-日内多', 'IM-日内空']
        project = 'IM-综合'
        horizontal_dict = {}
        
        for i in range(0, len(start_list)-1):
            demo = Tactic(start_date=start_list[i], end_date=end_list[i], project=project, all_name=self.all_name, name=None)
            df_all, df_all_cash = demo.all_member_data()
            df_all = df_all.sort_values(by = 'return_rank')
            market_index = market(project, self.start_date, self.end_date)
            market_index['index_return'] = market_index['close'].iloc[-1] / market_index['open'].iloc[0] - 1
            df_all['index_return'] = market_index['index_return'].iloc[0]
            df_all['收益率'] = df_all['收益率'].apply(lambda x: f'{x:.2%}')
            df_all['基准收益率'] = df_all['index_return'].apply(lambda x: f'{x:.2%}')
            df_all['排名'] = df_all['return_rank']  
            df_all = df_all[['排名', '收益率']]
            horizontal_dict[f'第{i+1}期'] = df_all

        # 初始化一个空字典来存储每个选手的排名数据  
        player_rankings = {}  
        
        # 遍历原始数据  
        for period, df in horizontal_dict.items():  
            # 只保留排名列  
            rank_df = df[['排名']]  
            # 遍历每个选手  
            for name in rank_df.index:  
                # 如果选手不在字典中，则创建一个新的DataFrame，以期数为索引  
                if name not in player_rankings:  
                    player_rankings[name] = pd.DataFrame(index=[period], columns=['排名'])  
                # 否则，将排名添加到现有DataFrame中  
                player_rankings[name].loc[period, '排名'] = rank_df.at[name, '排名']

        for key, value in player_rankings.items():  
            # 将结果存储回结果字典  
            value.reset_index(inplace=True, drop=False)
            value.rename(columns={'index': 'Period'}, inplace=True)
            player_rankings[key] = value.to_dict(orient='records')
        
        return player_rankings
    

    


######################################## 排行榜 #########################################
# 趋势日表现
    def trend_day(self):
        """
        趋势日表现
         ----------
        Returns:
            dataframe:
                index: 趋势日
                column: 进攻收益平均值、防守收益平均值、综合
        """
        
        
        # IM-多
        demo_long = Tactic(start_date='2024-5-13', end_date=end_list[-1], project='IM-日内多', all_name=self.all_name, name=None)
        return_long = demo_long.daily_return('IM-日内多')


        # IM-空
        demo_short = Tactic(start_date='2024-5-13', end_date=end_list[-1], project='IM-日内空', all_name=self.all_name, name=None)
        return_short = demo_short.daily_return('IM-日内空')
        
        
        df_trend = demo_long.whether_trend()
        df_trend = df_trend.loc[df_trend['whether_trend']==1]
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        trend_list = formatted_dates_str.to_list()   # 趋势日列表

        trend_long = return_long.loc[trend_list]
        trend_short = return_short.loc[trend_list]

        df_trend['attack'] = 0
        df_trend['defense'] = 0
        df_trend['trend'] = 'long'
        for index, row in df_trend.iterrows():
            if row['index_return'] > 0:
                df_trend.loc[index, 'attack'] = trend_long.loc[index]['每日平均收益率']
                df_trend.loc[index, 'defense'] = trend_short.loc[index]['每日平均收益率']
            else:
                df_trend.loc[index, 'attack'] = trend_short.loc[index]['每日平均收益率']
                df_trend.loc[index, 'defense'] = trend_long.loc[index]['每日平均收益率']
                df_trend.loc[index, 'trend'] = 'short'
        df_trend['compre'] = df_trend['attack'] + df_trend['defense']
        df_trend = df_trend[['attack', 'defense', 'return', 'trend']]
        return df_trend


   
    # 趋势排行榜
    def trend_rank(self):  
        """  
        趋势排行榜  
    
        Returns:  
            dataframe:  
                index: 选手名称  
                column: 每个趋势日的进攻收益  
        """  
        long_dict = {}
        short_dict = {}
        compre_dict = {}
        for i in range(0, len(start_list)-1):
            demo_long = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-日内多', all_name=self.all_name, name=None)  
            long_temp = demo_long.person_return() 
            for name, df in long_temp.items():
                if name not in long_dict:
                    long_dict[name] = df
                else:
                    long_dict[name] = pd.concat([long_dict[name], df])
                

            demo_short = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-日内空', all_name=self.all_name, name=None)  
            short_temp = demo_short.person_return() 
            for name, df in short_temp.items():
                if name not in short_dict:
                    short_dict[name] = df
                else:
                    short_dict[name] = pd.concat([short_dict[name], df])

            demo_compre = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-综合', all_name=self.all_name, name=None)  
            compre_temp = demo_compre.person_return()  
            for name, df in compre_temp.items():
                if df.empty:
                    continue
                if name not in compre_dict:
                    compre_dict[name] = df
                else:
                    compre_dict[name] = pd.concat([compre_dict[name], df])  
        
        demo_temp = Tactic(start_date=start_list[0], end_date=end_list[-1], project='IM-日内多', all_name=self.all_name, name=None)
        # 获取趋势日  
        df_trend = demo_temp.whether_trend()
        df_trend = df_trend.loc[df_trend['whether_trend']==1]
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        trend_list = formatted_dates_str.to_list()   # 趋势日列表 
    
        # 初始化结果DataFrame  
        result_df = pd.DataFrame(index=long_dict.keys(), columns=trend_list)  
    
        # 遍历每个选手的DataFrame  
        for name, df in long_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_df.at[name, trend_date] = 0  
        df_long_dict = {col: result_df[col] for col in result_df.columns}  
        for key, value in result_df.items():
            value = value.dropna()
            df_long_dict[key] = value

        # 初始化结果DataFrame  
        result_short_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
        # 遍历每个选手的DataFrame  
        for name, df in short_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_short_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_short_df.at[name, trend_date] = 0  
        df_short_dict = {col: result_short_df[col] for col in result_short_df.columns}  
        for key, value in result_short_df.items():
            value = value.dropna()
            df_short_dict[key] = value


        # 初始化结果DataFrame  
        result_compre_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
    
        # 遍历每个选手的DataFrame  
        for name, df in compre_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:   
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_compre_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_compre_df.at[name, trend_date] = 0  
        df_compre_dict = {col: result_compre_df[col] for col in result_compre_df.columns}  
        for key, value in result_compre_df.items():
            value = value.dropna()
            df_compre_dict[key] = value
        
        result_dict = {}
        for key in df_compre_dict.keys():  
            # 使用concat横向拼接DataFrame，并设置新的列名  
            df_combined = pd.concat([df_compre_dict[key], df_long_dict[key], df_short_dict[key]], axis=1)  
            # 重命名列  
            if df_trend.loc[key]['index_return'] > 0:
                df_combined.columns = ['综合', '进攻', '防守']  
            else:
                df_combined.columns = ['综合', '防守', '进攻']  
            # 将结果存储回结果字典  
            df_combined.reset_index(inplace=True, drop=False)
            df_combined.rename(columns={'index': 'Name'}, inplace=True)
            result_dict[key] = df_combined.to_dict(orient='records')
        return result_dict

    



    # 综合排行榜
    def day_rank(self):  
        """  
        综合排行榜  
    
        Returns:  
            dict:  
                index: 选手名称  
                column: 每天的收益  
        """  
        long_dict = {}
        short_dict = {}
        compre_dict = {}
        
        for i in range(0, len(start_list)):
            try:
                demo_long = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-日内多', all_name=self.all_name, name=None)  
                long_temp = demo_long.person_return() 
                for name, df in long_temp.items():
                    if name not in long_dict:
                        long_dict[name] = df
                    else:
                        long_dict[name] = pd.concat([long_dict[name], df])
                    

                demo_short = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-日内空', all_name=self.all_name, name=None)  
                short_temp = demo_short.person_return() 
                for name, df in short_temp.items():
                    if name not in short_dict:
                        short_dict[name] = df
                    else:
                        short_dict[name] = pd.concat([short_dict[name], df])

                demo_compre = Tactic(start_date=start_list[i], end_date=end_list[i], project='IM-综合', all_name=self.all_name, name=None)  
                compre_temp = demo_compre.person_return()  
                
                for name, df in compre_temp.items():
                    if df.empty:
                        continue 
                    if name not in compre_dict:
                        compre_dict[name] = df
                    else:
                        compre_dict[name] = pd.concat([compre_dict[name], df])  
            except Exception as e:
                print(e)

        
    
        demo_temp = Tactic(start_date=start_list[0], end_date=end_list[-1], project='IM-日内多', all_name=self.all_name, name=None)
        # 所有日期 
        df_trend = demo_temp.whether_trend()
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        trend_list = formatted_dates_str.to_list()   # 趋势日列表 
    
        # 初始化结果DataFrame  
        result_df = pd.DataFrame(index=long_dict.keys(), columns=trend_list)  
    
        # 遍历每个选手的DataFrame  
        for name, df in long_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_df.at[name, trend_date] = 0  
        df_long_dict = {col: result_df[col] for col in result_df.columns}  
        for key, value in result_df.items():
            value = value.dropna()
            df_long_dict[key] = value

        # 初始化结果DataFrame  
        result_short_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
        # 遍历每个选手的DataFrame  
        for name, df in short_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_short_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_short_df.at[name, trend_date] = 0  
        df_short_dict = {col: result_short_df[col] for col in result_short_df.columns}  
        for key, value in result_short_df.items():
            value = value.dropna()
            df_short_dict[key] = value


        # 初始化结果DataFrame  
        result_compre_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
    
        # 遍历每个选手的DataFrame  
        for name, df in compre_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:   
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_compre_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_compre_df.at[name, trend_date] = 0  
        df_compre_dict = {col: result_compre_df[col] for col in result_compre_df.columns}  
        for key, value in result_compre_df.items():
            value = value.dropna()
            df_compre_dict[key] = value
        result_dict = {}
        for key in df_compre_dict.keys():  
            # 使用concat横向拼接DataFrame，并设置新的列名  
            df_combined = pd.concat([df_compre_dict[key], df_long_dict[key], df_short_dict[key]], axis=1)  
            df_combined.columns = ['综合', 'IM-日内多', 'IM-日内空']  
            # 重命名列  
            # if df_trend.loc[key]['index_return'] > 0:
            #     df_combined.columns = ['综合', '进攻', '防守']  
            # else:
            #     df_combined.columns = ['综合', '防守', '进攻']  
            # 将结果存储回结果字典  
            df_combined.reset_index(inplace=True, drop=False)
            df_combined.rename(columns={'index': 'Name'}, inplace=True)
            result_dict[key] = df_combined.to_dict(orient='records')

        df = pd.DataFrame()
        for key, value in result_dict.items():
            value = pd.DataFrame(value)
            value['date'] = key
            df = pd.concat([df, value])
        df = df.fillna(0)
        filtered_df = df[~((df['综合'] == 0) & (df['IM-日内多'] == 0) & (df['IM-日内空'] == 0))]
        return filtered_df

       
    


    # 单日排行榜
    def single_radar(self, i):  
        """  
        综合排行榜  
    
        Returns:  
            dataframe:  
                index: 选手名称  
                column: 每天的收益  
        """  
        long_dict = {}
        short_dict = {}
        compre_dict = {}
        demo_long = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-日内多', all_name=self.all_name, name=None)  
        long_temp = demo_long.person_return() 
        for name, df in long_temp.items():
            if name not in long_dict:
                long_dict[name] = df
            else:
                long_dict[name] = pd.concat([long_dict[name], df])
                

        demo_short = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-日内空', all_name=self.all_name, name=None)  
        short_temp = demo_short.person_return() 
        for name, df in short_temp.items():
            if name not in short_dict:
                short_dict[name] = df
            else:
                short_dict[name] = pd.concat([short_dict[name], df])

        demo_compre = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-综合', all_name=self.all_name, name=None)  
        compre_temp = demo_compre.person_return()  
        df_compre, df_cash = demo_compre.all_member_data()
        for name, df in compre_temp.items():
            if df.empty:
                continue
            if name not in compre_dict:
                compre_dict[name] = df
            else:
                compre_dict[name] = pd.concat([compre_dict[name], df])  
        
        demo_temp = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-日内多', all_name=self.all_name, name=None)
        # 所有日期 
        df_trend = demo_temp.whether_trend()
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        trend_list = formatted_dates_str.to_list()   # 趋势日列表 
    
        # 初始化结果DataFrame  
        result_df = pd.DataFrame(index=long_dict.keys(), columns=trend_list)  
    
        # 遍历每个选手的DataFrame  
        for name, df in long_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_df.at[name, trend_date] = 0  
        df_long_dict = {col: result_df[col] for col in result_df.columns}  
        for key, value in result_df.items():
            value = value.dropna()
            df_long_dict[key] = value

        # 初始化结果DataFrame  
        result_short_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
        # 遍历每个选手的DataFrame  
        for name, df in short_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:  
                # 计算每个趋势日的平均收益率（如果需要）  
                # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_short_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_short_df.at[name, trend_date] = 0  
        df_short_dict = {col: result_short_df[col] for col in result_short_df.columns}  
        for key, value in result_short_df.items():
            value = value.dropna()
            df_short_dict[key] = value


        # 初始化结果DataFrame  
        result_compre_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
    
        # 遍历每个选手的DataFrame  
        for name, df in compre_dict.items():  
            # 筛选出趋势日的数据  
            trend_data = df.loc[df.index.isin(trend_list)]  
            # 假设'收益率'是DataFrame中的一列  
            if '收益率' in trend_data.columns:   
                for trend_date in trend_list:  
                    if trend_date in trend_data.index:  
                        result_compre_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                    else:  
                        # 如果没有该日期的数据，可以填充为NaN或其他值  
                        result_compre_df.at[name, trend_date] = 0  
        df_compre_dict = {col: result_compre_df[col] for col in result_compre_df.columns}  
        for key, value in result_compre_df.items():
            value = value.dropna()
            df_compre_dict[key] = value
        
        result_dict = {}
        for key in df_compre_dict.keys():  
            # 使用concat横向拼接DataFrame，并设置新的列名  
            df_combined = pd.concat([df_compre_dict[key], df_long_dict[key], df_short_dict[key]], axis=1)  
            # 重命名列  
            if df_trend.loc[key]['index_return'] > 0:
                df_combined.columns = ['综合', '进攻', '防守']  
            else:
                df_combined.columns = ['综合', '防守', '进攻']  
            result_dict[key] = df_combined

        # 合并所有DataFrame  
        all_data = pd.concat(result_dict.values())
        
        result ={}
        # # 分组并计算平均值  
        average_stats = all_data.groupby(by=all_data.index).sum() 
        average_stats[['盈亏比', '胜率', '稳定性']] = df_compre[['盈亏比', '盈利比例', '波动率']]
        average_stats['综合'] = df_compre['收益率'] * 100
        for column in average_stats.columns:
            average_stats[column] = average_stats[column].rank(ascending=True) / len(average_stats) * 100
        average_stats['稳定性'] = 100 - average_stats['稳定性']
        average_stats['信号数量'] = df_compre['交易次数']
        # 将结果存储回结果字典  
        average_stats.reset_index(inplace=True, drop=False)
        average_stats.rename(columns={'index': 'Name'}, inplace=True)
        result[f'第{i}期'] = average_stats.to_dict(orient='records')
        return result


    ## 部门排行榜
    def Rank_department(self):
        person_rank = self.trend_rank()
        department_dict = {
            '数据部': ['谢佳钊', '肖炎辉', '范创', '高昌盛', '徐晓朋', '庄斯杰'],
            '投资部': ['张炳坤', '刘亚洲', '伍水明', '杨宝圣', '肖健', '谭泽松'],
            '总经办-技术部': ['章魏康', '钟远金', '唐硕', '陈广', '赵明昊', '刘俊浩', '何博轩', '陈楷锐', '郭总', '黄永朗', '黄梓聪']
        }
        # 新字典，用于存储按部门分类的人员数据  
        new_dict = {}  
        
        # 遍历日期字典  
        for date, people in person_rank.items():  
            # 初始化部门数据字典  
            department_data = {}  
            
            # 遍历每个人员  
            for person in people:  
                name = person['Name']  
                
                # 检查该人员是否属于某个部门  
                for dept, members in department_dict.items():  
                    if name in members:  
                        # 如果属于某个部门，则将该人员数据添加到该部门的字典中  
                        if dept not in department_data:  
                            department_data[dept] = []  
                        department_data[dept].append(person)  
                        break  # 找到后跳出循环，避免重复添加  
            
            # 将该日期的部门数据添加到新字典中  
            new_dict[date] = department_data  
        
        return new_dict
        

######################################## 趋势日数据 ########################################
    def trend_data(self):
        result_dict = {}
        Period_list = []
        # 获取趋势日  
        demo_temp = Tactic(start_date=start_list[0], end_date=end_list[-1], project='IM-日内多', all_name=self.all_name, name=None)
        df_trend = demo_temp.whether_trend()
        df_trend = df_trend.loc[df_trend['whether_trend']==1]
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        df_trend['return'] = (df_trend['close'] / df_trend['open'] - 1) * 100
        df_trend['amplitude'] = (df_trend['high'] - df_trend['low']) / df_trend['pre_close'] * 100
        # df_trend['amplitude'] = (df_trend['high'] - df_trend['low']) / df_trend['open'] * 100
        df_trend = df_trend[['return', 'amplitude']]
        for date in df_trend.index:
            period = check_period(start_list=start_list, end_list=end_list, target_date=date)
            Period_list.append(period)
        df_trend['period'] = Period_list
        df_trend.reset_index(inplace=True, drop=False)
        df_trend.rename(columns={'index': 'Date'}, inplace=True)
        # result_dict = df_trend.to_dict(orient='records')
        return df_trend
    

    ## IM走势图
    def trend_trend(self):
        result_dict = {}
        Period_list = []
        # 获取趋势日  
        demo_temp = Tactic(start_date=start_list[0], end_date=end_list[-1], project='IM-日内多', all_name=self.all_name, name=None)
        df_trend = demo_temp.whether_trend()
        df_trend = df_trend.loc[df_trend['whether_trend']==1]
        formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
        formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
        df_trend.index = formatted_dates_str
        df_trend['return'] = df_trend['close'] - df_trend['pre_close']
        df_trend['amplitude'] = df_trend['high'] - df_trend['low']
        print(df_trend.index)
        close_dict = {}
        for date in formatted_dates:
            detail = ths.history_future(symbol='IM', start_time=date, end_time=date, period='1m', df=True)[['eob', 'close']]
            detail = detail.to_dict(orient='records')
            close_dict[date] = detail
        
        # 转换为只包含时间和价格的列表，因为原始需求只提到了价格  
        processed_time_series = {  
            date: [{''.join(time_price['eob'].split(' ')[1:]): time_price['close']} for time_price in data]  
            for date, data in close_dict.items()  
        }

        # 最终结果字典  
        result_dict = []  
        
        for index, row in df_trend.iterrows():  
            date = row['trade_date'].isoformat()  # 转换为'YYYY-MM-DD'格式  
            daily_data = {  
                '日期': date,  
                '涨幅': row['return'],  
                '振幅': row['amplitude'],  
                '分时': processed_time_series.get(date, [])  # 获取对应的分时数据，如果没有则为空列表  
            }  
            result_dict.append(daily_data)
        return result_dict
    


    # 所有参赛选手及参加期数
    def all_person(self):
        """
        所有参赛选手及参加期数
                
        """
        # project_list = ['IF-日内多', 'IF-日内空', 'IM-日内多', 'IM-日内空']
        project = 'IM-综合'
        horizontal_dict = {}
        
        for i in range(0, len(start_list)-1):
            demo = Tactic(start_date=start_list[i], end_date=end_list[i], project=project, all_name=self.all_name, name=None)
            df_all, df_all_cash = demo.all_member_data()
            df_all = df_all.sort_values(by = 'return_rank')
            market_index = market(project, self.start_date, self.end_date)
            market_index['index_return'] = market_index['close'].iloc[-1] / market_index['open'].iloc[0] - 1
            df_all['index_return'] = market_index['index_return'].iloc[0]
            df_all['收益率'] = df_all['收益率'].apply(lambda x: f'{x:.2%}')
            df_all['基准收益率'] = df_all['index_return'].apply(lambda x: f'{x:.2%}')
            df_all['排名'] = df_all['return_rank']  
            df_all = df_all[['排名', '收益率']]
            horizontal_dict[i+1] = df_all

        # 初始化一个空字典来存储每个选手的排名数据  
        player_rankings = {}  
        
        # 遍历原始数据  
        for period, df in horizontal_dict.items():  
            # 只保留排名列  
            rank_df = df[['排名']]  
            # 遍历每个选手  
            for name in rank_df.index:  
                # 如果选手不在字典中，则创建一个新的DataFrame，以期数为索引  
                if name not in player_rankings:  
                    player_rankings[name] = pd.DataFrame(index=[period], columns=['排名'])  
                # 否则，将排名添加到现有DataFrame中  
                player_rankings[name].loc[period, '排名'] = rank_df.at[name, '排名']

        result_list = []  
        for key, value in player_rankings.items():  
            
            # # 将结果存储回结果字典  
            # value.reset_index(inplace=True, drop=False)
            # value.rename(columns={'index': 'Period'}, inplace=True)
            # player_rankings[key] = value.to_dict(orient='records')
            person_dict = {'name': key, 'periods': value.index.tolist()}
            result_list.append(person_dict)
        
        return result_list
    




    ################### test ######################
    # 单日排行榜
    @lru_cache
    def single_radar_test(self):  
        """  
        综合排行榜  
    
        Returns:  
            dataframe:  
                index: 选手名称  
                column: 每天的收益  
        """  
        result ={}
        for i in range(1, len(start_list)):
            long_dict = {}
            short_dict = {}
            compre_dict = {}
            demo_long = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-日内多', all_name=self.all_name, name=None)  
            long_temp = demo_long.person_return() 
            for name, df in long_temp.items():
                if name not in long_dict:
                    long_dict[name] = df
                else:
                    long_dict[name] = pd.concat([long_dict[name], df])
                    

            demo_short = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-日内空', all_name=self.all_name, name=None)  
            short_temp = demo_short.person_return() 
            for name, df in short_temp.items():
                if name not in short_dict:
                    short_dict[name] = df
                else:
                    short_dict[name] = pd.concat([short_dict[name], df])

            demo_compre = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-综合', all_name=self.all_name, name=None)  
            compre_temp = demo_compre.person_return()  
            df_compre, df_cash = demo_compre.all_member_data()
            for name, df in compre_temp.items():
                if df.empty:
                    continue
                if name not in compre_dict:
                    compre_dict[name] = df
                else:
                    compre_dict[name] = pd.concat([compre_dict[name], df])  
            
            demo_temp = Tactic(start_date=start_list[i-1], end_date=end_list[i-1], project='IM-日内多', all_name=self.all_name, name=None)
            # 所有日期 
            df_trend = demo_temp.whether_trend()
            formatted_dates = pd.to_datetime(df_trend['trade_date'], format='%Y%m%d')
            formatted_dates_str = formatted_dates.dt.strftime('%Y-%m-%d')
            df_trend.index = formatted_dates_str
            trend_list = formatted_dates_str.to_list()   # 趋势日列表 
        
            # 初始化结果DataFrame  
            result_df = pd.DataFrame(index=long_dict.keys(), columns=trend_list)  
        
            # 遍历每个选手的DataFrame  
            for name, df in long_dict.items():  
                # 筛选出趋势日的数据  
                trend_data = df.loc[df.index.isin(trend_list)]  
                # 假设'收益率'是DataFrame中的一列  
                if '收益率' in trend_data.columns:  
                    # 计算每个趋势日的平均收益率（如果需要）  
                    # 这里我假设直接取数据，不计算平均值（因为可能是日收益率）  
                    # 如果需要平均值，可以使用：trend_data['收益率'].mean() 但这里按日期处理  
                    for trend_date in trend_list:  
                        if trend_date in trend_data.index:  
                            result_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                        else:  
                            # 如果没有该日期的数据，可以填充为NaN或其他值  
                            result_df.at[name, trend_date] = 0  
            df_long_dict = {col: result_df[col] for col in result_df.columns}  
            for key, value in result_df.items():
                value = value.dropna()
                df_long_dict[key] = value

            # 初始化结果DataFrame  
            result_short_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
            # 遍历每个选手的DataFrame  
            for name, df in short_dict.items():  
                # 筛选出趋势日的数据  
                trend_data = df.loc[df.index.isin(trend_list)]  
                # 假设'收益率'是DataFrame中的一列  
                if '收益率' in trend_data.columns:  
                    for trend_date in trend_list:  
                        if trend_date in trend_data.index:  
                            result_short_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                        else:  
                            # 如果没有该日期的数据，可以填充为NaN或其他值  
                            result_short_df.at[name, trend_date] = 0  
            df_short_dict = {col: result_short_df[col] for col in result_short_df.columns}  
            for key, value in result_short_df.items():
                value = value.dropna()
                df_short_dict[key] = value


            # 初始化结果DataFrame  
            result_compre_df = pd.DataFrame(index=short_dict.keys(), columns=trend_list)  
        
            # 遍历每个选手的DataFrame  
            for name, df in compre_dict.items():  
                # 筛选出趋势日的数据  
                trend_data = df.loc[df.index.isin(trend_list)]  
                # 假设'收益率'是DataFrame中的一列  
                if '收益率' in trend_data.columns:   
                    for trend_date in trend_list:  
                        if trend_date in trend_data.index:  
                            result_compre_df.at[name, trend_date] = trend_data.loc[trend_date, '收益率']  
                        else:  
                            # 如果没有该日期的数据，可以填充为NaN或其他值  
                            result_compre_df.at[name, trend_date] = None  
            df_compre_dict = {col: result_compre_df[col] for col in result_compre_df.columns}  
            for key, value in result_compre_df.items():
                value = value.dropna()
                df_compre_dict[key] = value
            
            result_dict = {}
            for key in df_compre_dict.keys():  
                # 使用concat横向拼接DataFrame，并设置新的列名  
                df_combined = pd.concat([df_compre_dict[key], df_long_dict[key], df_short_dict[key]], axis=1)  
                # 重命名列  
                if df_trend.loc[key]['index_return'] > 0:
                    df_combined.columns = ['综合', '进攻', '防守']  
                else:
                    df_combined.columns = ['综合', '防守', '进攻']  
                result_dict[key] = df_combined

            # 合并所有DataFrame  
            all_data = pd.concat(result_dict.values())
            
            
            # # 分组并计算平均值  
            average_stats = all_data.groupby(by=all_data.index).sum() 
            average_stats[['盈亏比', '胜率', '稳定性']] = df_compre[['盈亏比', '盈利比例', '波动率']]
            average_stats['综合'] = df_compre['收益率'] * 100
            for column in average_stats.columns:
                average_stats[column] = average_stats[column].rank(ascending=True) / len(average_stats) * 100
            average_stats['稳定性'] = 100 - average_stats['稳定性']
            average_stats['信号数量'] = df_compre['交易次数']
            average_stats['综合排名'] = average_stats['综合'].rank(ascending=False)
            # 将结果存储回结果字典  
            # average_stats.reset_index(inplace=True, drop=False)
            # average_stats.rename(columns={'index': 'Name'}, inplace=True)
            result[i] = average_stats
        # 遍历原始数据  
        final_result = {}
        for period, df in result.items():  
            # 对于df中的每一行（即每个人）  
            for name, row in df.iterrows():  
                # 如果这个人在结果字典中不存在，则初始化一个空的DataFrame  
                if name not in final_result:  
                    final_result[name] = pd.DataFrame(columns=['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性', '信号数量', '综合排名'], index=[])  
                # 将当前期的数据作为一个新行添加到对应人的DataFrame中  
                # 注意：这里我们通过重设索引来确保期数作为新行的索引  
                temp_df = pd.DataFrame([row.tolist()], index=[period], columns=row.index)  
                # 使用concat来合并，axis=0表示纵向合并（按行）  
                final_result[name] = pd.concat([final_result[name], temp_df], axis=0) 
        return final_result
    

    def radar(self, name):
        result = {}
        person_df = self.single_radar_test()[name]
        # 直接计算每列的平均值  
        avg_series = person_df.mean(numeric_only=True)  
        
        # 或者，如果你想要一个具有相同列名的DataFrame  
        avg_df = person_df.mean(numeric_only=True).to_frame().T  
        avg_df.columns = person_df.columns  
        result = avg_df.to_dict(orient='records')
        result[0]['Name'] = name
        return result


    def attack(self, name, i):
        result = {}
        person_df = self.single_radar_test()[name].loc[i]
        result = person_df.to_dict()
        result['Name'] = name
        result['period'] = i
        return result
    

        
    def fetch_player_data(self, name):  
        """  
        根据选手名称从MySQL数据库中读取该选手表的所有数据  
    
        :param name: 选手名称，假设数据库中的表名与选手名称相同  
        :return: 选手表的所有数据列表，每个元素是一个包含一行数据的元组  
        """  
        try:  
            # 连接到MySQL数据库  
            connection = pymysql.connect(  
                host='192.168.10.210',  # 例如: 'localhost'  
                user='root',  # 数据库用户名  
                password='djct003',  # 数据库密码  
                database='tactic_data',  # 数据库名  
                charset='utf8mb4',  # 字符集，支持emoji等  
                cursorclass=pymysql.cursors.DictCursor  # 使用字典游标，使返回的结果为字典格式（可选）  
            )  
    
            if connection.open:  
                with connection.cursor() as cursor:  
                    # 构造SQL查询语句  
                    sql_query = f"SELECT * FROM `{name}`"  
                    
                    cursor.execute(sql_query)  
                    
                    # 获取所有记录  
                    records = cursor.fetchall()  
                    
                    # 如果使用了DictCursor，则records中的每个元素都是字典；否则是元组  
                    return records  
        except pymysql.MySQLError as e:  
            print(f"Error: {e}")  
            return None  
        finally:  
            # 无论是否发生异常，都尝试关闭连接  
            connection.close()  