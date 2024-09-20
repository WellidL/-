import json
import pandas as pd
from flask import Flask, request
import Z_Read 
from flask_cors import CORS
from decimal import Decimal
from datetime import datetime,date
from flask_caching import Cache
import numpy as np
import new_data_source as ds
import tushare as ts
from DjData import ths
pro = ts.pro_api('818670fa68bc204c217143cdb75efeae1986031841ff8ca2c6a855bd')
ts.set_token('818670fa68bc204c217143cdb75efeae1986031841ff8ca2c6a855bd')



# 创建Flask服务
app = Flask(__name__)
# 设置缓存配置
app.config['CACHE_TYPE'] = 'simple'  # 也可以使用其他类型，如 redis, filesystem 等
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600  # 缓存超时时间为1小时
cache = Cache(app)
#开启跨域处理
CORS(app)

############################################# 比赛数据 #######################################
def get_match():
    df = Z_Read.date_data()
    start_list = df['start_date'].tolist()
    end_list = df['end_date'].tolist()

    return start_list, end_list


############################################# 辅助函数 #######################################
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


def filter_by_time(df, start_date, end_date):
    
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



def transfer_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')  
    formatted_date = date_obj.strftime('%Y%m%d')  
    return formatted_date

########################################################################################### 接口 #####################################################################################

############################################# 个人数据 #######################################

## 个人指标雷达图
@app.route('/day_in/Radar', methods=['GET'])
# @cache.memoize(timeout=3600)
def Radar():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Radar?name=陈楷锐
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    # try:
    final_dict = {}
    # 判断入参是否为空
    if request.args is None:
        return_dict['return_code'] = '504'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    
    # 获取传入的参数
    get_data_pramers = request.args.to_dict()
    name = get_data_pramers.get('name')

    # data = Z_Read.fetch_player_data(name)
    data = Z_Read.fetch_player_data2()
    data = pd.DataFrame(data)
    data = data[data['name'] == name]
    person_df = pd.DataFrame(data)   
    person_df = person_df[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性']]

    # 或者，如果你想要一个具有相同列名的DataFrame  
    avg_df = person_df.mean().to_frame().T  
    avg_df.columns = person_df.columns 
    avg_df = avg_df[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性']]
    avg_df.rename(columns={'综合': 'compre', '进攻': 'attack', '防守': 'defence', '盈亏比': 'profitLoss', '胜率': 'winRate', '稳定性': 'stability'}, inplace=True)
    #加权评分，总和为100%,进攻占比20%，防守占比20%，胜率占比15%，收益均比占比10%，稳定性占比15%，综合收益占比20%
    avg_df['total'] = avg_df['attack'] * 0.2 + avg_df['defence'] * 0.2 + avg_df['winRate'] * 0.15 + avg_df['compre'] * 0.20 + avg_df['stability'] * 0.15 + avg_df['profitLoss'] * 0.1
    result = avg_df.to_dict(orient='records')
    
    final_dict['name'] = name
    final_dict['data'] = result
    return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(final_dict)
    return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
    # except:
    #     return_dict['code'] = 400
    #     return_dict['msg'] = '请求失败'
    #     return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)



# ## 与第一名的对比
# @app.route('/day_in/Compare', methods=['GET'])
# @cache.memoize(timeout=3600)
# def Compare():
#     """
#     功能：
#         返回个人指标


#     接口访问示例：
#     http://192.168.10.237:8881/day_in/Compare?name=伍水明
#     """
# # 默认返回内容
#     return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
#     try:
#         final_dict = {}
#         # 判断入参是否为空
#         if request.args is None:
#             return_dict['return_code'] = '504'
#             return_dict['return_info'] = '请求参数为空'
#             return json.dumps(return_dict, ensure_ascii=False)
        

#         ## 获得所有人数据
#         data = Z_Read.fetch_daily_data()
#         data = pd.DataFrame(data)
#         name_list = data['Name'].unique().tolist()
#         df_list = []
#         # 获取传入的参数
#         get_data_pramers = request.args.to_dict()
#         name = get_data_pramers.get('name')
#         for man in name_list:
#             data_1 = Z_Read.fetch_player_data(man)
#             data_1 = pd.DataFrame(data_1)
#             avg_df = data_1.mean().to_frame().T  
#             try:
#                 avg_df.columns = data_1.columns 
#                 avg_df = avg_df[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性', '信号数量']]
#                 avg_df.rename(columns={'综合': 'compre', '进攻': 'attack', '防守': 'defence', '盈亏比': 'profitLoss', '胜率': 'winRate', '稳定性': 'stability', '信号数量': 'signal'}, inplace=True)
#                 avg_df['compre'] = avg_df['compre'] * 2
#                 avg_df['name'] = man
#                 df_list.append(avg_df)
#             except Exception as e:
#                 print(f'{man}没有数据, {e}')
#         total_df = pd.concat(df_list, axis=0)
#         total_df.sort_values(by='compre', ascending=False, inplace=True)
#         number_one = total_df.iloc[0]
#         other = total_df[total_df['name'] == name].T
#         final_df = pd.concat([number_one, other], axis=1).T
#         final_dict = final_df.to_dict(orient='records')
#         return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(final_dict)
#         return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
        
#     except:
#         return_dict['code'] = 400
#         return_dict['msg'] = '请求失败'
#         return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


## 与第一名的对比2
@app.route('/day_in/Compare', methods=['GET'])
# @cache.memoize(timeout=3600)
def Compare():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Compare?name=伍水明
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        final_dict = {}
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
    

  
        get_data_pramers = request.args.to_dict()
        name = get_data_pramers.get('name')
        data = Z_Read.fetch_player_data2()
        data = pd.DataFrame(data)
        name_list = data['name'].unique().tolist()
        grouped = data.groupby('name')
        group_list = []
        for man, group in grouped:
            group = group[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性', '信号数量']]
            group = group.mean().to_frame().T
            group['name'] = man
            group_list.append(group)
        
        total_df = pd.concat(group_list, axis=0)
        total_df = total_df[['name', '综合', '进攻', '防守', '盈亏比', '胜率', '稳定性', '信号数量']]
        total_df.rename(columns={'综合': 'compre', '进攻': 'attack', '防守': 'defence', '盈亏比': 'profitLoss', '胜率': 'winRate', '稳定性': 'stability', '信号数量': 'signal'}, inplace=True)
        total_df.sort_values(by='compre', ascending=False, inplace=True)
        number_one = total_df.iloc[0]
        other = total_df[total_df['name'] == name].T
        final_df = pd.concat([number_one, other], axis=1).T
        final_dict = final_df.to_dict(orient='records')
        return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(final_dict)
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
            
    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


# ## 进攻能力和排名
# @app.route('/day_in/Attack', methods=['GET'])
# # @cache.memoize(timeout=3600)
# def Attack():
#     """
#     功能：
#         返回个人指标


#     接口访问示例：
#     http://192.168.10.237:8881/day_in/Attack?name=章魏康&i=5
#     """
# # 默认返回内容
#     return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
#     try:
#         final_dict = {}
#         # 判断入参是否为空
#         if request.args is None:
#             return_dict['return_code'] = '504'
#             return_dict['return_info'] = '请求参数为空'
#             return json.dumps(return_dict, ensure_ascii=False)
        
#         # 获取传入的参数
#         get_data_pramers = request.args.to_dict()
#         name = get_data_pramers.get('name')
#         i = get_data_pramers.get('i')
#         # data = Z_Read.fetch_player_data(name)
#         # person_df = pd.DataFrame(data)   
#         data = Z_Read.fetch_player_data2()
#         person_df = pd.DataFrame(data)   
#         person_df = person_df[person_df['name'] == name]
#         person_df = person_df[['期数', '综合', '进攻', '防守', '盈亏比', '胜率', '稳定性']]
#         person_df.rename(columns={'期数': 'period', '综合': 'compre', '进攻': 'attack', '防守': 'defence', '盈亏比': 'profitLoss', '胜率': 'winRate', '稳定性': 'stability'}, inplace=True)
#         person_df = person_df[person_df['period'] == int(i)]
#         result = person_df.iloc[0].to_dict()
#         final_dict['name'] = name
#         final_dict['data'] = [result]
#         return_dict['result'] = final_dict
#         return json.dumps(return_dict, ensure_ascii=False)
    
#     except:
#         return_dict['code'] = 400
#         return_dict['msg'] = '请求失败'
#         return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

## 进攻能力和排名2
@app.route('/day_in/Attack', methods=['GET'])
def Attack():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Attack?name=陈楷锐
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        final_dict = {}
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取传入的参数
        get_data_pramers = request.args.to_dict()
        name = get_data_pramers.get('name')

        # data = Z_Read.fetch_player_data(name)
        data = Z_Read.fetch_player_data2()
        data = pd.DataFrame(data)
        grouped = data.groupby('name')
        df_list = []
        for man, person_df in grouped:
            
            person_df = person_df[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性']]
            # 或者，如果你想要一个具有相同列名的DataFrame  
            avg_df = person_df.mean().to_frame().T  
            avg_df.columns = person_df.columns 
            avg_df = avg_df[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性']]
            avg_df.rename(columns={'综合': 'compre', '进攻': 'attack', '防守': 'defence', '盈亏比': 'profitLoss', '胜率': 'winRate', '稳定性': 'stability'}, inplace=True)
            #加权评分，总和为100%,进攻占比20%，防守占比20%，胜率占比15%，收益均比占比10%，稳定性占比15%，综合收益占比20%
            avg_df['total'] = avg_df['attack'] * 0.2 + avg_df['defence'] * 0.2 + avg_df['winRate'] * 0.15 + avg_df['compre'] * 0.20 + avg_df['stability'] * 0.15 + avg_df['profitLoss'] * 0.1
            
            avg_df['name'] = man
            df_list.append(avg_df)
        total_df = pd.concat(df_list, axis=0)
        total_df['rank'] = total_df['total'].rank(method='min', ascending=False)
        single_df = total_df[total_df['name'] == name]
        single_df['intensity'] = 0 if single_df['attack'].iloc[0] > 50 else 1
        single_df = single_df[['rank', 'intensity']]
        
        
        result = single_df.to_dict(orient='records')
        
        final_dict['name'] = name
        final_dict['data'] = result
        return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(final_dict)
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

## 排名波动情况
@app.route('/day_in/Rank_change', methods=['GET'])
# @cache.memoize(timeout=3600)
def Rank_change():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Rank_change?name=章魏康
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取传入的参数
        get_data_pramers = request.args.to_dict()
        name = get_data_pramers.get('name')
        # data = Z_Read.fetch_player_data(name)

        data = Z_Read.fetch_player_data2()
        person_df = pd.DataFrame(data)   
        person_df = person_df[person_df['name'] == name]

        result = {}
        temp_list = []
        for index, row in person_df.iterrows():
            temp = {}
            temp['period'] = row['期数']
            temp['rank'] = row['综合排名']
            temp_list.append(temp)
        result['name'] = name
        result['data'] = temp_list
        return_dict['result'] = result
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
    

## 个人趋势日表现
@app.route('/day_in/Trend_person', methods=['GET'])
# @cache.memoize(timeout=3600)
def Trend_person():
    """
    功能：
        返回个人指标
        进攻、防守、综合

    接口访问示例：
    http://192.168.10.237:8881/day_in/Trend_person?name=章魏康
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)

        df = pd.DataFrame()
        # 获取传入的参数
        get_data_pramers = request.args.to_dict()
        name = get_data_pramers.get('name')
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        trend_data = Z_Read.trend_data()
        trend_data = pd.DataFrame(trend_data)
        date_list = trend_data['trade_date'].unique().tolist()
        data = data[data['date'].isin(date_list)]
        data = data[data['Name'] == name]
        df = data[['综合', 'IM-日内多', 'IM-日内空']]
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([trend_data, df], axis=1)
        df['进攻'] = np.where(df['return'] > 0, df['IM-日内多'], df['IM-日内空'])  
        df['防守'] = np.where(df['return'] > 0, df['IM-日内空'], df['IM-日内多']) 
        df = df[['return', 'amplitude', '综合', '进攻', '防守']]
        df['compreReturn'] = df['综合'] * 2
        df['compreRatio'] = df['综合'] / df['amplitude']
        df['attackReturn'] = df['进攻'] 
        df['attackRatio'] = df['进攻'] / df['amplitude']
        df['defenseReturn'] = df['防守'] 
        df['defenseRatio'] = df['防守'] / df['amplitude']
        df = df[['return', 'amplitude', 'compreReturn', 'attackReturn', 'defenseReturn', 'compreRatio', 'attackRatio', 'defenseRatio']]
        date_change_list = []
        for date in date_list:
            date_obj = datetime.strptime(date, "%Y-%m-%d")  
            timestamp = date_obj.timestamp()  
            date_change_list.append(timestamp)
        
        total_list = []
        for i in range(0, len(date_change_list)):
            single_result = {}
            single_result['date'] = date_change_list[i]
            single_result['data'] = Z_Read.NanEncoder()._nan_to_none(df.iloc[i].to_dict())
            total_list.append(single_result)
        return_dict['result'] = total_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

## 个人信息
## 个人信息
@app.route('/day_in/Info', methods=['GET'])
def Info():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.210:8881/day_in/Info?person=李陇杰
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    # try:
    final_dict = {}
    # 判断入参是否为空
    if request.args is None:
        return_dict['return_code'] = '504'
        return_dict['return_info'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)
    
    # 获取传入的参数
    get_data_pramers = request.args.to_dict()
    person = get_data_pramers.get('person')

    # data = Z_Read.fetch_player_data(name)
    data = Z_Read.fetch_player_data2()
    data = pd.DataFrame(data)
    grouped = data.groupby('name')
    df_list = []
    for man, person_df in grouped:
        
        person_df = person_df[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性']]
        # 或者，如果你想要一个具有相同列名的DataFrame  
        avg_df = person_df.mean().to_frame().T  
        avg_df.columns = person_df.columns 
        avg_df = avg_df[['综合', '进攻', '防守', '盈亏比', '胜率', '稳定性']]
        avg_df.rename(columns={'综合': 'compre', '进攻': 'attack', '防守': 'defense', '盈亏比': 'profitLoss', '胜率': 'winRate', '稳定性': 'stability'}, inplace=True)
        #加权评分，总和为100%,进攻占比20%，防守占比20%，胜率占比15%，收益均比占比10%，稳定性占比15%，综合收益占比20%
        avg_df['total'] = avg_df['attack'] * 0.2 + avg_df['defense'] * 0.2 + avg_df['winRate'] * 0.15 + avg_df['compre'] * 0.20 + avg_df['stability'] * 0.15 + avg_df['profitLoss'] * 0.1
        
        avg_df['name'] = man
        df_list.append(avg_df)
    total_df = pd.concat(df_list, axis=0)
    total_df['rank'] = total_df['total'].rank(method='min', ascending=False)
    single_df = total_df[total_df['name'] == person]
    single_df['o_intensity'] = '弱' if single_df['attack'].iloc[0] > 50 else '强'
    single_df['d_intensity'] = '弱' if single_df['defense'].iloc[0] > 50 else '强'


    df_day = pd.read_excel('获奖数据.xlsx', engine='openpyxl', sheet_name='day')
    df_five = pd.read_excel('获奖数据.xlsx', engine='openpyxl', sheet_name='five')
    df_day = df_day.sort_values(by='period', ascending=True)
    df_five = df_five.sort_values(by='period', ascending=True)
    

    day_copy = df_day.copy()[['period', 'name', 'rank']]
    five_copy = df_five.copy()[['period', 'name', 'rank']]
    day_copy.rename(columns={'period': 'dayPeriod', 'name': 'dayName', 'rank': 'dayRank'}, inplace=True)
    five_copy.rename(columns={'period': 'fivePeriod', 'name': 'fiveName', 'rank': 'fiveRank'}, inplace=True)
    df_five_day = pd.concat([day_copy, five_copy], axis=1)
    day_rank_1 = df_five_day[df_five_day['dayRank'] == 1][['dayName', 'dayPeriod']]
    day_rank_2 = df_five_day[df_five_day['dayRank'] == 2][['dayName', 'dayPeriod']]
    five_rank_1 = df_five_day[df_five_day['fiveRank'] == 1][['fiveName', 'fivePeriod']]
    five_rank_2 = df_five_day[df_five_day['fiveRank'] == 2][['fiveName', 'fivePeriod']]
    day_rank = pd.merge(day_rank_1, day_rank_2, on='dayPeriod', how='left')
    five_rank = pd.merge(five_rank_1, five_rank_2, on='fivePeriod', how='left')

    if day_rank['dayPeriod'].iloc[-1] // 2 != 0:
        repeated_index = np.repeat(five_rank.index, 2)    
        result_df = pd.concat([five_rank, five_rank], ignore_index=True) if five_rank.shape[0] > 0 else five_rank  
        five_rank = result_df.sort_values(by='fivePeriod', ascending=True)

    else:
        repeated_index = np.repeat(five_rank.index, 2)    
        result_df = pd.concat([five_rank, five_rank], ignore_index=True) if five_rank.shape[0] > 0 else five_rank  
        five_rank = result_df.sort_values(by='fivePeriod', ascending=True)
        # 创建一个全NaN的Series，其索引与DataFrame的列名相匹配  
        empty_row = pd.Series([np.nan] * len(five_rank.columns), index=five_rank.columns)  
        # 将这个空行追加到DataFrame的末尾  
        five_rank = result_df.append(empty_row, ignore_index=True)  # 如果不需要重置索引，可以去掉ignore_index=True  

    day_rank.rename(columns={'dayName_x': 'dayFirst', 'dayName_y': 'daySecond'}, inplace=True)
    five_rank.rename(columns={'fiveName_x': 'fiveFirst', 'fiveName_y': 'fiveSecond'}, inplace=True)
    five_rank = five_rank.sort_values(by='fivePeriod', ascending=True).reset_index(drop=True)
    df = pd.concat([day_rank, five_rank], axis=1)
    df_todict = df.to_dict('records')
    form = {'reward': df_todict}
    day_mix = pd.concat([df_day, df_five], axis=0)
    day_mix = day_mix.dropna()
    name_list = day_mix['name'].unique().tolist()
    day_in_list = []
    five_list = []
    sum_list = []
    money_list = []
    for name in name_list:
        dayin_num = len(day_mix[(day_mix['name']==name) & (day_mix['project']=='日内')])
        five_num = len(day_mix[(day_mix['name']==name) & (day_mix['project']=='5日')])
        sum_num = dayin_num + five_num
        money = day_mix[day_mix['name']==name]['money'].sum()
        day_in_list.append(dayin_num)
        five_list.append(five_num)
        sum_list.append(sum_num)
        money_list.append(money)

    total_data = pd.DataFrame({
        'name': name_list,
        'dayin': day_in_list,
        'five': five_list,
        'sum': sum_list,
        'money': money_list
    })
    total_data = total_data[total_data['name'] != None]
    try:
        person_df = total_data[total_data['name'] == person]
        reward_num = person_df['sum'].iloc[0]
    except:
        reward_num = 0
    single_df = single_df[['total','rank', 'o_intensity', 'd_intensity']]
    
    info = f"综合评分: {single_df['total'].iloc[0]}\n综合排名: {single_df['rank'].iloc[0]}\n进攻能力: {single_df['o_intensity'].iloc[0]}\n防守能力: {single_df['d_intensity'].iloc[0]}\n获奖次数: {reward_num}\n"
    
    
    
    final_dict['name'] = person
    final_dict['data'] = info
    return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(final_dict)
    return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
    # except:
    #     return_dict['code'] = 400
    #     return_dict['msg'] = '请求失败'
    #     return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)




## 个人信号图
@app.route('/day_in/Personal_signal', methods=['GET'])
# @cache.memoize(timeout=3600)
def Personal_signal():
    """
    功能：
        返回个人指标
        进攻、防守、综合

    接口访问示例：
    http://192.168.10.237:8881/day_in/Personal_signal?name=章魏康&project=IM-日内多

    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)

        df = pd.DataFrame()
        # 获取传入的参数
        get_data_pramers = request.args.to_dict()
        name = get_data_pramers.get('name')
        get_data_pramers = request.args.to_dict()
        project = get_data_pramers.get('project')

        df = Z_Read.all_data()
        IM_df = df[df['project'].str.contains('IM')]
        if project == 'IM-日内多':
            
            IM_df = IM_df[((IM_df['project'] == 'IM-日内多') & (IM_df['交易类型'].isin(['开多', '平多'])))]
        else:
            IM_df = IM_df[((IM_df['project'] == 'IM-日内空') & (IM_df['交易类型'].isin(['开空', '平空'])))] 
  
        single_data = IM_df[IM_df['name'] == name]
        single_data = single_data.sort_values(by='委托时间', ascending=True)
        single_data = single_data.reset_index(drop=True)
        single_data = single_data[['委托时间', '委托数量']]
        
        single_data.rename(columns={'委托时间': 'operateTime', '委托数量': 'operate'}, inplace=True)
        single_data['operateTime'] = pd.to_datetime(single_data['operateTime'], errors='coerce').dt.tz_localize('Asia/Shanghai').apply(lambda x: x.timestamp())
        single_dict = single_data.to_dict(orient='records')

        return_dict['result'] = single_dict
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
    

## 个人信息


##################################################### 整体 ##################################################

########################## 收益 ################################
## 整体收益
@app.route('/day_in/Return_whole', methods=['GET'])
# @cache.memoize(timeout=3600)
def Return_whole():
    """
    功能：
        返回整体每日收益
        进攻、防守、综合

    接口访问示例：
    http://192.168.10.237:8881/day_in/Return_whole
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        date_change_list = []
        df = pd.DataFrame()
        total_list = []
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        
        date_list = data['date'].unique().tolist()
        reference = pro.fut_daily(ts_code='IM.CFX', start_date=transfer_date(date_list[0]), end_date=transfer_date(date_list[-1])).sort_values('trade_date', ascending=True)
        # reference = ths.history_future(symbol='IM', start_time='2024-05-10', end_time=date_list[-1], period='1d', df=True)[['eob', 'close']]
        # 当日期货涨幅
        reference['change'] = (reference['change1'] / reference['pre_close']) * 100
        reference = reference[['change']]
        reference = (1+reference['change']*0.01).cumprod()
        reference = (reference-1)*100
        reference = reference.reset_index(drop=True)
        
        
        grouped = data.groupby('date')
        df_list = []
        for date, temp in grouped:
            temp = temp[['综合', 'IM-日内多', 'IM-日内空']]
            temp = temp.mean()
            temp = pd.DataFrame(temp).T
            df_list.append(temp)
        df_total = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_total = (1+df_total*0.01).cumprod()
        df_total = (df_total-1)*100
        df_total = df_total.reset_index(drop=True)
        df_total = pd.concat([df_total, reference], axis=1)
        df_total.rename(columns={'综合': 'compreReturn', 'IM-日内多': 'dayLong', 'IM-日内空': 'dayShort'}, inplace=True)
        df_total['compreReturn'] = df_total['compreReturn'] * 2
        for date in date_list:
            date_obj = datetime.strptime(date, "%Y-%m-%d")  
            timestamp = date_obj.timestamp()  
            date_change_list.append(timestamp)
        for i in range(0, len(df_total)):
            single_result = {}
            single_result['date'] = date_change_list[i]
            single_result['data'] = Z_Read.NanEncoder()._nan_to_none(df_total.iloc[i].to_dict())
            total_list.append(single_result)
        return_dict['result'] = total_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)



# ## 分组收益 （要修改）
# @app.route('/day_in/Return_group', methods=['GET'])
# # # @cache.memoize(timeout=3600) 
# def Return_group():
#     """
#     功能：
#         返回分组收益  
        

#     接口访问示例：
#     http://192.168.10.237:8881/day_in/Return_group
#     """
# # 默认返回内容
#     return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
#     try:
#         # 判断入参是否为空
#         if request.args is None:
#             return_dict['return_code'] = '504'
#             return_dict['return_info'] = '请求参数为空'
#             return json.dumps(return_dict, ensure_ascii=False)
        
#         date_change_list = []
#         df = pd.DataFrame()
#         total_list = []
#         period_list = []
#         data = Z_Read.fetch_daily_data()
#         data = pd.DataFrame(data)
#         date_list = data['date'].unique().tolist()
#         name_list = data['Name'].unique().tolist()
#         for name in name_list:
#             period_data = Z_Read.fetch_player_data(name)
#             period_data = pd.DataFrame(period_data)
#             period_data['Name'] = name
#             period_list.append(period_data)
#         period_data = pd.concat(period_list, axis=0).reset_index(drop=True)
#         group_period = period_data.groupby('期数')


#         group_1_list = []
#         group_2_list = []
#         group_3_list = []
#         group_4_list = []
#         group_5_list = []
#         for group_period, temp in group_period:
#             temp['综合'] = temp['综合'] * 2
#             temp = temp.sort_values('综合', ascending=False)
#             num_list = group_people(len(temp))
#             group_1 = temp.iloc[:num_list[0]]['Name']
#             group_2 = temp.iloc[num_list[0]:num_list[0]+num_list[1]]['Name']
#             group_3 = temp.iloc[num_list[0]+num_list[1]:num_list[0]+num_list[1]+num_list[2]]['Name']
#             group_4 = temp.iloc[num_list[0]+num_list[1]+num_list[2]:num_list[0]+num_list[1]+num_list[2]+num_list[3]]['Name']
#             group_5 = temp.iloc[num_list[0]+num_list[1]+num_list[2]+num_list[3]:]['Name']
#             group_1_list.append(group_1)
#             group_2_list.append(group_2)
#             group_3_list.append(group_3)
#             group_4_list.append(group_4)
#             group_5_list.append(group_5)

#         start_list, end_list = get_match()
#         return_list1 = []
#         return_list2 = []
#         return_list3 = []
#         return_list4 = []
#         return_list5 = []
#         for i in range(0, len(start_list)-2):
#             temp_data = filter_by_time(data, start_list[i], end_list[i])
            
#             group_1_mean = temp_data[temp_data['Name'].isin(group_1_list[i])]
#             grouped_1 = group_1_mean.groupby('date')
#             group_1_return = grouped_1['综合'].mean()
#             return_list1.append(group_1_return)
        

#             group_2_mean = temp_data[temp_data['Name'].isin(group_2_list[i])]
#             grouped_2 = group_2_mean.groupby('date')
#             group_2_return = grouped_2['综合'].mean()
#             return_list2.append(group_2_return)

#             group_3_mean = temp_data[temp_data['Name'].isin(group_3_list[i])]
#             grouped_3 = group_3_mean.groupby('date')
#             group_3_return = grouped_3['综合'].mean()
#             return_list3.append(group_3_return)

#             group_4_mean = temp_data[temp_data['Name'].isin(group_4_list[i])]
#             grouped_4 = group_4_mean.groupby('date')
#             group_4_return = grouped_4['综合'].mean()
#             return_list4.append(group_4_return)

#             group_5_mean = temp_data[temp_data['Name'].isin(group_5_list[i])]
#             grouped_5 = group_5_mean.groupby('date')
#             group_5_return = grouped_5['综合'].mean()
#             return_list5.append(group_5_return)

#         df_1 = pd.concat(return_list1, axis=0).reset_index()
#         df_1 = df_1[['综合']]
#         df_1.rename(columns={'综合': 'groupOne'}, inplace=True)
#         df_2 = pd.concat(return_list2, axis=0).reset_index()
#         df_2 = df_2[['综合']]
#         df_2.rename(columns={'综合': 'groupTwo'}, inplace=True)
#         df_3 = pd.concat(return_list3, axis=0).reset_index()
#         df_3 = df_3[['综合']]
#         df_3.rename(columns={'综合': 'groupThree'}, inplace=True)
#         df_4 = pd.concat(return_list4, axis=0).reset_index()
#         df_4 = df_4[['综合']]
#         df_4.rename(columns={'综合': 'groupFour'}, inplace=True)
#         df_5 = pd.concat(return_list5, axis=0).reset_index()
#         df_5 = df_5[['综合']]
#         df_5.rename(columns={'综合': 'groupFive'}, inplace=True)
#         total_data = pd.concat([df_1, df_2, df_3, df_4, df_5], axis=1)
#         for date in date_list[:-4]:
#                 date_obj = datetime.strptime(date, "%Y-%m-%d")  
#                 timestamp = date_obj.timestamp()  
#                 date_change_list.append(timestamp)
        
#         total_data['date'] = date_change_list
#         total_data = total_data[['groupOne', 'groupTwo', 'groupThree', 'groupFour', 'groupFive']]
#         total_data = (1+total_data*0.01).cumprod()
#         total_data = (total_data-1)*100
#         total_data = total_data.reset_index(drop=True)
#         for i in range(0, len(total_data)):
#             single_result = {}
#             single_result['date'] = date_change_list[i]
#             single_result['data'] = Z_Read.NanEncoder()._nan_to_none(total_data.iloc[i].to_dict())
#             total_list.append(single_result)    

#         return_dict['result'] = total_list
#         return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

#     except:
#         return_dict['code'] = 400
#         return_dict['msg'] = '请求失败'
#         return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)



## 分组收益2
@app.route('/day_in/Return_group', methods=['GET'])
# # @cache.memoize(timeout=3600) 
def Return_group():
    """
    功能：
        返回分组收益  
        

    接口访问示例：
    http://192.168.10.237:8881/day_in/Return_group
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        date_change_list = []
        total_list = []
        date_list = data['date'].unique().tolist()
        # df = pd.DataFrame()
        # 
        # period_list = []
        # data = Z_Read.fetch_daily_data()
        # data = pd.DataFrame(data)
        
        # name_list = data['Name'].unique().tolist()
        # for name in name_list:
        #     period_data = Z_Read.fetch_player_data(name)
        #     period_data = pd.DataFrame(period_data)
        #     period_data['Name'] = name
        #     period_list.append(period_data)
        # period_data = pd.concat(period_list, axis=0).reset_index(drop=True)
        period_data = Z_Read.fetch_player_data2()
        period_data = pd.DataFrame(period_data)
        group_period = period_data.groupby('期数')


        group_1_list = []
        group_2_list = []
        group_3_list = []
        group_4_list = []
        group_5_list = []
        for group_period, temp in group_period:
            temp['综合'] = temp['综合'] * 2
            temp = temp.sort_values('综合', ascending=False)
            num_list = group_people(len(temp))
            group_1 = temp.iloc[:num_list[0]]['name']
            group_2 = temp.iloc[num_list[0]:num_list[0]+num_list[1]]['name']
            group_3 = temp.iloc[num_list[0]+num_list[1]:num_list[0]+num_list[1]+num_list[2]]['name']
            group_4 = temp.iloc[num_list[0]+num_list[1]+num_list[2]:num_list[0]+num_list[1]+num_list[2]+num_list[3]]['name']
            group_5 = temp.iloc[num_list[0]+num_list[1]+num_list[2]+num_list[3]:]['name']
            group_1_list.append(group_1)
            group_2_list.append(group_2)
            group_3_list.append(group_3)
            group_4_list.append(group_4)
            group_5_list.append(group_5)

        start_list, end_list = get_match()
        return_list1 = []
        return_list2 = []
        return_list3 = []
        return_list4 = []
        return_list5 = []
        for i in range(0, len(start_list)-1):
            temp_data = filter_by_time(data, start_list[i], end_list[i])
            
            group_1_mean = temp_data[temp_data['Name'].isin(group_1_list[i])]
            grouped_1 = group_1_mean.groupby('date')
            group_1_return = grouped_1['综合'].mean()
            return_list1.append(group_1_return)
        

            group_2_mean = temp_data[temp_data['Name'].isin(group_2_list[i])]
            grouped_2 = group_2_mean.groupby('date')
            group_2_return = grouped_2['综合'].mean()
            return_list2.append(group_2_return)

            group_3_mean = temp_data[temp_data['Name'].isin(group_3_list[i])]
            grouped_3 = group_3_mean.groupby('date')
            group_3_return = grouped_3['综合'].mean()
            return_list3.append(group_3_return)

            group_4_mean = temp_data[temp_data['Name'].isin(group_4_list[i])]
            grouped_4 = group_4_mean.groupby('date')
            group_4_return = grouped_4['综合'].mean()
            return_list4.append(group_4_return)

            group_5_mean = temp_data[temp_data['Name'].isin(group_5_list[i])]
            grouped_5 = group_5_mean.groupby('date')
            group_5_return = grouped_5['综合'].mean()
            return_list5.append(group_5_return)

        df_1 = pd.concat(return_list1, axis=0).reset_index()
        df_1 = df_1[['综合']]
        df_1.rename(columns={'综合': 'groupOne'}, inplace=True)
        df_2 = pd.concat(return_list2, axis=0).reset_index()
        df_2 = df_2[['综合']]
        df_2.rename(columns={'综合': 'groupTwo'}, inplace=True)
        df_3 = pd.concat(return_list3, axis=0).reset_index()
        df_3 = df_3[['综合']]
        df_3.rename(columns={'综合': 'groupThree'}, inplace=True)
        df_4 = pd.concat(return_list4, axis=0).reset_index()
        df_4 = df_4[['综合']]
        df_4.rename(columns={'综合': 'groupFour'}, inplace=True)
        df_5 = pd.concat(return_list5, axis=0).reset_index()
        df_5 = df_5[['综合']]
        df_5.rename(columns={'综合': 'groupFive'}, inplace=True)
        total_data = pd.concat([df_1, df_2, df_3, df_4, df_5], axis=1)
        for date in date_list:
                date_obj = datetime.strptime(date, "%Y-%m-%d")  
                timestamp = date_obj.timestamp()  
                date_change_list.append(timestamp)


        # total_data['date'] = date_change_list
        total_data = total_data[['groupOne', 'groupTwo', 'groupThree', 'groupFour', 'groupFive']]
        total_data = (1+total_data*0.01).cumprod()
        total_data = (total_data-1)*100
        total_data = total_data.reset_index(drop=True)
        for i in range(0, len(total_data)):
            single_result = {}
            single_result['date'] = date_change_list[i]
            single_result['data'] = Z_Read.NanEncoder()._nan_to_none(total_data.iloc[i].to_dict())
            total_list.append(single_result)    

        return_dict['result'] = total_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


## 部门收益
@app.route('/day_in/Return_department', methods=['GET'])
# # @cache.memoize(timeout=3600) 
def Return_department():
    """
    功能：
        返回分组收益  
        

    接口访问示例：
    http://192.168.10.237:8881/day_in/Return_department
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        date_change_list = []
        df = pd.DataFrame()
        total_list = []
        period_list = []
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        date_list = data['date'].unique().tolist()
        department_dict = {
            '数据部': ['谢佳钊', '肖炎辉', '范创', '高昌盛', '徐晓朋', '庄斯杰'],
            '投资部': ['张炳坤', '刘亚洲', '伍水明', '杨宝圣', '肖健', '谭泽松'],
            '软件部': ['郭文彬', '李陇杰', '孔德枞', '罗嘉俊', '屈中翔', '阮云泉', '陶松', '阳俊'],
            '开发部': ['林秋宇', '欧佰康', '孙彪', '游凯文', '张晓光', '周思捷'],
            '总经办-技术部': ['章魏康', '钟远金', '唐硕', '陈广', '赵明昊', '刘俊浩', '何博轩', '陈楷锐', '郭婉婷', '郭总', '黄永朗', 
                        '黄梓聪', '赖晓梅', '罗威', '庞优华', '王畅', '张湘子', '张紫荷', '郑妮斯', '黄慧仪', ]
        }
        grouped = data.groupby('date')
        data_list = []
        invest_list = []
        software_list = []
        develop_list = []
        tech_list = []
        for date, temp in grouped:
            df_data = temp.loc[temp['Name'].isin(department_dict['数据部'])]['综合'].mean()
            df_invest = temp.loc[temp['Name'].isin(department_dict['投资部'])]['综合'].mean()
            df_software = temp.loc[temp['Name'].isin(department_dict['软件部'])]['综合'].mean()
            df_develop = temp.loc[temp['Name'].isin(department_dict['开发部'])]['综合'].mean()
            df_tech = temp.loc[temp['Name'].isin(department_dict['总经办-技术部'])]['综合'].mean()
            data_list.append(df_data)
            invest_list.append(df_invest)
            software_list.append(df_software)
            develop_list.append(df_develop)
            tech_list.append(df_tech)
        total_data = pd.DataFrame({
            'record': data_list,
            'invest': invest_list,
            'software': software_list,
            'develop': develop_list,
            'tech': tech_list
        })
        for date in date_list:
                date_obj = datetime.strptime(date, "%Y-%m-%d")  
                timestamp = date_obj.timestamp()  
                date_change_list.append(timestamp)
        
        total_data['date'] = date_change_list
        total_data = total_data[['record', 'invest', 'software', 'develop', 'tech']]
        total_data = (1+total_data*0.01).cumprod()
        total_data = (total_data-1)*100
        total_data = total_data.reset_index(drop=True)
        for i in range(0, len(total_data)):
            single_result = {}
            single_result['date'] = date_change_list[i]
            single_result['data'] = Z_Read.NanEncoder()._nan_to_none(total_data.iloc[i].to_dict())
            total_list.append(single_result)    

        return_dict['result'] = total_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)



def modify_data(df):
# 初始化一些变量来追踪流程状态
    in_process = False  # 用来追踪是否在流程中
    for i in range(len(df)):
        if df.loc[i, 'operate'] == 1:
            if in_process:
                # 如果已经在流程中，出现额外的1，将其改为0
                df.loc[i, 'operate'] = 0
            else:
                # 标记流程开始
                in_process = True
        elif df.loc[i, 'operate'] == 2:
            if in_process:
                # 正常流程结束
                in_process = False
            else:
                # 如果流程没有开始就结束，将其改为0
                df.loc[i, 'operate'] = 0
    return df

## 以信号为信号的策略
@app.route('/day_in/Signal', methods=['GET'])
# @cache.memoize(timeout=3600)
def Signal():
    """
    功能：
        以信号为信号的策略

    接口访问示例：
    http://192.168.10.237:8881/day_in/Signal?ratio=15&min=20&project=IM-日内多
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        # 获取参数
        get_data_pramers = request.args.to_dict()
        ratio = get_data_pramers.get('ratio')
        ratio = float(ratio)
        get_data_pramers = request.args.to_dict()
        min = get_data_pramers.get('min')
        min = int(min)
        get_data_pramers = request.args.to_dict()
        project = get_data_pramers.get('project')


        # df_dict = Z_Read.data_get()
        # result_df = pd.DataFrame()  

        # # 遍历字典  
        # df_list = []
        # for name, df in df_dict.items():  
        #     # 为当前DataFrame添加一列来标识原始名字  
        #     df['name'] = name  
        #     # 将修改后的DataFrame追加到结果DataFrame中  
        #     df_list.append(df)  
        # df = pd.concat(df_list)
        # # df = df[(df['project'] == 'IM-日内多') | (df['project'] == 'IM-日内空')]

    
        # df = df[(df['project'] == 'IM-日内多') | (df['project'] == 'IM-日内空')]
        df = Z_Read.all_data()
        df = df[(df['project'] == project)]
        df['委托时间'] = pd.to_datetime(df['委托时间'])  

        # 计算每个时间点的分组键（这里是每十分钟分组的时间戳）  
        # 使用 dt.floor('10T') 来向下取整到最近的十分钟  
        group_key_timestamps = df['委托时间'].dt.floor(f'{min}T')  
        
        # 将分组键作为新列添加到 DataFrame 中  
        df['分组时间戳'] = group_key_timestamps  
        # 示例：打印每个分组的键（这些键是 datetime 对象）  
        name_list = df['name'].unique().tolist()

        grouped = df.groupby('分组时间戳')  


        longOpen_list = []
        shortOpen_list = []
        longClose_list = []
        shortClose_list = []
        time_list = []
        for time, temp in grouped:
            if project == 'IM-日内多':
                longOpen = len(temp[temp['交易类型'] == '开多'])
                longClose = len(temp[temp['交易类型'] == '平多'])

            else:
                longOpen = len(temp[temp['交易类型'] == '开空'])
                longClose = len(temp[temp['交易类型'] == '平空'])

        df = pd.DataFrame({
            'longOpen': longOpen_list,
            # 'shortOpen': shortOpen_list,
            'longClose': longClose_list,
            # 'shortClose': shortClose_list,
            'time': time_list
        })
        time_list2 = []
        for time in df['time']:
            time = datetime.fromtimestamp(time)
            time = time.strftime('%Y-%m-%d')
            time_list2.append(time)
        df['day'] = time_list2
        df['participant_num'] = 0
        start_list, end_list = get_match()
        

        def set_operate(row, ratio):
            if row['score'] >= ratio:
                return 1
            elif row['score'] <= -ratio:
                return 2
            else:
                return 0

        total_df = Z_Read.fetch_player_data2()
        total_df = pd.DataFrame(total_df)
        total_df = total_df[['期数']]
        for i in range(0, len(start_list)-1):
            df.loc[(df['day'] >= start_list[i]) & (df['day'] <= end_list[i]), 'period'] = i+1
            # df.loc[df['period'] == i+1]['participant_num'] = len(total_df[total_df['期数'] == i+1])
        df['participant_num'] = df['period'].apply(lambda x: len(total_df[total_df['期数'] == x]) if pd.notna(x) else None) 
        df.reset_index(drop=True, inplace=True)
        # df['noOperate'] = df['participant_num'] - df['longOpen'] - df['shortClose']- df['shortOpen'] - df['longClose']
        df['noOperate'] = df['participant_num'] - df['longOpen']  - df['longClose']
        # df = df[['time', 'longOpen', 'shortClose', 'shortOpen', 'longClose', 'noOperate', 'participant_num']]
        df = df[['time', 'longOpen', 'longClose', 'noOperate', 'participant_num']]
        df['longOpen_grade'] = df['longOpen'] * 2
        # df['shortClose'] = df['shortClose'] / df['participant_num']
        # df['shortOpen'] = df['shortOpen'] / df['participant_num']
        df['longClose_grade'] = df['longClose'] * (-2)
        df['noOperate'] = df['noOperate'] * 0 
        df['score'] = df['longOpen_grade'] + df['longClose_grade'] + df['noOperate']
        df['operate'] = df.apply(lambda row: set_operate(row, ratio), axis=1)  
        df = modify_data(df)
        result = df.to_dict(orient='records')
        
        return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(result)
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


########################### 排行榜 ###############################
# ## 综合排行榜
# @app.route('/day_in/Compre_rank', methods=['GET'])
# # @cache.memoize(timeout=3600)
# def Compre_rank():
#     """
#     功能：
#         返回个人指标


#     接口访问示例：
#     http://192.168.10.237:8881/day_in/Compre_rank?project=day
#     """
# # 默认返回内容
#     return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
#     try:
#         # 判断入参是否为空
#         if request.args is None:
#             return_dict['return_code'] = '504'
#             return_dict['return_info'] = '请求参数为空'
#             return json.dumps(return_dict, ensure_ascii=False)
        
#         # 获取传入的参数
#         get_data_pramers = request.args.to_dict()
#         project = get_data_pramers.get('project')
#         if project == 'day':
#             # 读取日内数据
#             data = Z_Read.fetch_daily_data()
#             data = pd.DataFrame(data)
#         grouped = data.groupby('Name')
#         df_list = []
        
#         for name, temp in grouped:
#             single_result = {}
#             df = temp[['综合', 'IM-日内多', 'IM-日内空']]
#             df.rename(columns={'综合': 'compre', 'IM-日内多': 'dayLong', 'IM-日内空': 'dayShort'}, inplace=True)
#             df['compre'] = df['compre'] * 2

#             df = df.sum()
#             df = pd.DataFrame(df).T
#             df_to_dict = df.to_dict('records')
#             single_result['name'] = name
#             single_result['data'] = Z_Read.NanEncoder()._nan_to_none(df_to_dict)
#             df_list.append(single_result)

#         return_dict['result'] = df_list
#         return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

#     except:
#         return_dict['code'] = 400
#         return_dict['msg'] = '请求失败'
#         return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

## 综合排行榜
@app.route('/day_in/Compre_rank', methods=['GET'])
# @cache.memoize(timeout=3600)
def Compre_rank():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Compre_rank?project=day
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取传入的参数
        get_data_pramers = request.args.to_dict()
        project = get_data_pramers.get('project')
        if project == 'day':
            # 读取日内数据
            data = Z_Read.period_data()
            
        grouped = data.groupby('name')
        df_list = []
        
        for name, temp in grouped:
            single_result = {}
            df = temp[['综合revenue', 'IM-日内多revenue', 'IM-日内空revenue']]
            df.rename(columns={'综合revenue': 'compre', 'IM-日内多revenue': 'dayLong', 'IM-日内空revenue': 'dayShort'}, inplace=True)

            df = df.sum()
            df = pd.DataFrame(df).T
            df_to_dict = df.to_dict('records')
            single_result['name'] = name
            single_result['data'] = Z_Read.NanEncoder()._nan_to_none(df_to_dict)
            df_list.append(single_result)

        return_dict['result'] = df_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


## 趋势排行榜
@app.route('/day_in/Trend_rank', methods=['GET'])
# @cache.memoize(timeout=3600)
def Trend_rank():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Trend_rank
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
                return_dict['return_code'] = '504'
                return_dict['return_info'] = '请求参数为空'
                return json.dumps(return_dict, ensure_ascii=False)
        
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        trend_data = Z_Read.trend_data()
        trend_data = pd.DataFrame(trend_data)
        date_list = trend_data['trade_date'].unique().tolist()
        data = data[data['date'].isin(date_list)]
        grouped = data.groupby('date')
        temp_list = []
        for date, temp in grouped:
            if trend_data[trend_data['trade_date'] == date]['return'].values[0] > 0:
                temp.rename(columns={'Name': 'name', '综合': 'compre', 'IM-日内多': 'attack', 'IM-日内空': 'defense'}, inplace=True)
            else:
                temp.rename(columns={'Name': 'name', '综合': 'compre', 'IM-日内多': 'defense', 'IM-日内空': 'attack'}, inplace=True)
            temp['compre'] = temp['compre'] * 2
            temp_list.append(temp)
        data = pd.concat(temp_list)
        
        grouped_name = data.groupby('name')
        df_list = []
        for name, temp in grouped_name:
            single_result = {}
            df = temp[['compre', 'attack', 'defense']]
            df = df.sum()
            df = pd.DataFrame(df).T
            df['name'] = name
            
            df_list.append(df)
        total_df = pd.concat(df_list)
        result = total_df.to_dict('records')
        return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(result)
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


## 获奖排行榜
@app.route('/day_in/Reward_rank', methods=['GET'])
# @cache.memoize(timeout=3600)
def Reward_rank():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Reward_rank
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
    # 判断入参是否为空
        if request.args is None:
                return_dict['return_code'] = '504'
                return_dict['return_info'] = '请求参数为空'
                return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取传入的参数
        get_data_pramers = request.args.to_dict()
        project = get_data_pramers.get('project')
        
        df_day = pd.read_excel('获奖数据.xlsx', engine='openpyxl', sheet_name='day')
        df_five = pd.read_excel('获奖数据.xlsx', engine='openpyxl', sheet_name='five')
        df_day = df_day.sort_values(by='period', ascending=True)
        df_five = df_five.sort_values(by='period', ascending=True)
        

        day_copy = df_day.copy()[['period', 'name', 'rank']]
        five_copy = df_five.copy()[['period', 'name', 'rank']]
        day_copy.rename(columns={'period': 'dayPeriod', 'name': 'dayName', 'rank': 'dayRank'}, inplace=True)
        five_copy.rename(columns={'period': 'fivePeriod', 'name': 'fiveName', 'rank': 'fiveRank'}, inplace=True)
        df_five_day = pd.concat([day_copy, five_copy], axis=1)
        day_rank_1 = df_five_day[df_five_day['dayRank'] == 1][['dayName', 'dayPeriod']]
        day_rank_2 = df_five_day[df_five_day['dayRank'] == 2][['dayName', 'dayPeriod']]
        five_rank_1 = df_five_day[df_five_day['fiveRank'] == 1][['fiveName', 'fivePeriod']]
        five_rank_2 = df_five_day[df_five_day['fiveRank'] == 2][['fiveName', 'fivePeriod']]
        day_rank = pd.merge(day_rank_1, day_rank_2, on='dayPeriod', how='left')
        five_rank = pd.merge(five_rank_1, five_rank_2, on='fivePeriod', how='left')

        if day_rank['dayPeriod'].iloc[-1] // 2 != 0:
            repeated_index = np.repeat(five_rank.index, 2)    
            result_df = pd.concat([five_rank, five_rank], ignore_index=True) if five_rank.shape[0] > 0 else five_rank  
            five_rank = result_df.sort_values(by='fivePeriod', ascending=True)

        else:
            repeated_index = np.repeat(five_rank.index, 2)    
            result_df = pd.concat([five_rank, five_rank], ignore_index=True) if five_rank.shape[0] > 0 else five_rank  
            five_rank = result_df.sort_values(by='fivePeriod', ascending=True)
            # 创建一个全NaN的Series，其索引与DataFrame的列名相匹配  
            empty_row = pd.Series([np.nan] * len(five_rank.columns), index=five_rank.columns)  
            # 将这个空行追加到DataFrame的末尾  
            five_rank = result_df.append(empty_row, ignore_index=True)  # 如果不需要重置索引，可以去掉ignore_index=True  

        day_rank.rename(columns={'dayName_x': 'dayFirst', 'dayName_y': 'daySecond'}, inplace=True)
        five_rank.rename(columns={'fiveName_x': 'fiveFirst', 'fiveName_y': 'fiveSecond'}, inplace=True)
        five_rank = five_rank.sort_values(by='fivePeriod', ascending=True).reset_index(drop=True)
        df = pd.concat([day_rank, five_rank], axis=1)
        df_todict = df.to_dict('records')
        form = {'reward': df_todict}
   
        day_mix = pd.concat([df_day, df_five], axis=0)
        day_mix = day_mix.dropna()
        name_list = day_mix['name'].unique().tolist()
        day_in_list = []
        five_list = []
        sum_list = []
        money_list = []
        for name in name_list:
            dayin_num = len(day_mix[(day_mix['name']==name) & (day_mix['project']=='日内')])
            five_num = len(day_mix[(day_mix['name']==name) & (day_mix['project']=='5日')])
            sum_num = dayin_num + five_num
            money = day_mix[day_mix['name']==name]['money'].sum()
            day_in_list.append(dayin_num)
            five_list.append(five_num)
            sum_list.append(sum_num)
            money_list.append(money)

        total_data = pd.DataFrame({
            'name': name_list,
            'dayin': day_in_list,
            'five': five_list,
            'sum': sum_list,
            'money': money_list
        })
        total_data = total_data[total_data['name'] != '无']
        total_data = total_data.sort_values(by='money', ascending=False)
        total_data['rank'] = range(1, len(total_data) + 1)
        total_dict = total_data.to_dict('records')
        form['rewardRank'] = total_dict
        return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(form)
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)


################################################# 趋势 #################################################
## 整体趋势日表现
@app.route('/day_in/Trend_whole', methods=['GET'])
# @cache.memoize(timeout=3600)
def Trend_whole():
    """
    功能：
        返回整体指标
        进攻、防守、综合

    接口访问示例：
    http://192.168.10.237:8881/day_in/Trend_whole
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        date_change_list = []
        df = pd.DataFrame()
        total_list = []
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        trend_data = Z_Read.trend_data()
        trend_data = pd.DataFrame(trend_data)
        date_list = trend_data['trade_date'].unique().tolist()
        data = data[data['date'].isin(date_list)]
        grouped = data.groupby('date')
        df_list = []
        for date, temp in grouped:
            temp = temp[['综合', 'IM-日内多', 'IM-日内空']]
            temp = temp.mean()
            temp = pd.DataFrame(temp).T
            df_list.append(temp)
        df_total = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_total = pd.concat([trend_data, df_total], axis=1)
        df_total['进攻'] = np.where(df_total['return'] > 0, df_total['IM-日内多'], df_total['IM-日内空'])
        df_total['防守'] = np.where(df_total['return'] > 0, df_total['IM-日内空'], df_total['IM-日内多'])
        df_total = df_total[['return', 'amplitude', '综合', '进攻', '防守']]
        df_total['进攻收益占比'] = df_total['进攻'] / df_total['amplitude']
        df_total.rename(columns={'涨幅': 'return', '振幅': 'amplitude', '综合': 'compreReturn', '进攻': 'attackReturn', '防守': 'defenseReturn', '进攻收益占比': 'attackRatio'}, inplace=True)
        df_total['compreReturn'] = df_total['compreReturn'] * 2

        for date in date_list:
            date_obj = datetime.strptime(date, "%Y-%m-%d")  
            timestamp = date_obj.timestamp()  
            date_change_list.append(timestamp)
        for i in range(0, len(df_total)):
            single_result = {}
            single_result['date'] = date_change_list[i]
            single_result['data'] = Z_Read.NanEncoder()._nan_to_none(df_total.iloc[i].to_dict())
            total_list.append(single_result)
        return_dict['result'] = total_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)




## 收益分布情况
@app.route('/day_in/Return_distribution', methods=['GET'])
# @cache.memoize(timeout=3600)
def Return_distribution():
    """
    功能：
        返回整体指标
        进攻、防守、综合

    接口访问示例：
    http://192.168.10.237:8881/day_in/Return_distribution?date=2024-08-05
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取参数
        get_data_pramers = request.args.to_dict()
        date = get_data_pramers.get('date')
        date_change_list = []
        df = pd.DataFrame()
        total_list = []
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        data = data[data['date'] == date]
        trend_data = Z_Read.trend_data()
        trend_data = pd.DataFrame(trend_data)
        trend_data = trend_data[trend_data['trade_date'] == date]
        if trend_data.iloc[0]['return'] > 0:
            data.rename(columns={'IM-日内多': 'attack', 'IM-日内空': 'defense'}, inplace=True)
        else:
            data.rename(columns={'IM-日内多': 'defense', 'IM-日内空': 'attack'}, inplace=True)
        
        # df_index = pro.fut_daily(ts_code='IM.CFX', start_date=transfer_date(date), end_date=transfer_date(date))
        # # 当日期货涨幅
        # change = (df_index['close'] / df_index['pre_close'] - 1).iloc[0] * 100
        # swing = (df_index['high'].iloc[0] - df_index['low'].iloc[0]) / df_index['pre_close'].iloc[0] * 100
        df_index = pro.fut_daily(ts_code='IM.CFX', start_date=transfer_date(date), end_date=transfer_date(date)).sort_values('trade_date', ascending=True).reset_index(drop=True)
        change = (df_index['change1'] / df_index['pre_close'] - 1).iloc[0] * 100
        swing = (df_index['high'].iloc[0] - df_index['low'].iloc[0]) / df_index['pre_close'].iloc[0] * 100


        # 参赛人数
        num = len(data)
        top = len(data[data['attack'] > swing*2*0.8])
        middle = len(data[(data['attack'] < swing*2*0.8) & (data['attack'] > swing*2*0.6)])
        down = num - top - middle
        df_comp = pd.DataFrame({
                'overEighty': [top],
                'sixtytoEighty': [middle],
                'belowSixty': [down]
            })
        df_dict = df_comp.to_dict(orient='records')
        date_stamp = datetime.strptime(date, "%Y-%m-%d")  
        timestamp = date_stamp.timestamp()
        return_dict['result'] = {'date': timestamp, 'data': df_dict}

        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)



## 排行榜
@app.route('/day_in/Rank', methods=['GET'])
# @cache.memoize(timeout=3600)
def Rank():
    """
    功能：
        返回特定趋势日个人的情况

    接口访问示例：
    http://192.168.10.237:8881/day_in/Rank?date=2024-08-05
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取参数
        get_data_pramers = request.args.to_dict()
        date = get_data_pramers.get('date')
        date_change_list = []
        df = pd.DataFrame()
        total_list = []
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        data = data[data['date'] == date]
        data = data[['Name', '综合', 'IM-日内多', 'IM-日内空']]
        data.rename(columns={'Name': 'name', '综合': 'compre', 'IM-日内多': 'dayLong', 'IM-日内空': 'dayShort'}, inplace=True)
        data['compre'] = data['compre'] * 2
        df_dict = data.to_dict(orient='records')
        date_stamp = datetime.strptime(date, "%Y-%m-%d")  
        timestamp = date_stamp.timestamp()
        return_dict['result'] = {'date': timestamp, 'data': df_dict}
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)



## 部门排行榜
@app.route('/day_in/Rank_department', methods=['GET'])
# # @cache.memoize(timeout=3600) 
def Rank_department():
    """
    功能：
        返回分组收益  
        

    接口访问示例：
    http://192.168.10.237:8881/day_in/Rank_department?date=2024-08-05
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取参数
        get_data_pramers = request.args.to_dict()
        date = get_data_pramers.get('date')

        date_change_list = []
        df = pd.DataFrame()
        total_list = []
        period_list = []
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        date_list = data['date'].unique().tolist()
        department_dict = {
            '数据部': ['谢佳钊', '肖炎辉', '范创', '高昌盛', '徐晓朋', '庄斯杰'],
            '投资部': ['张炳坤', '刘亚洲', '伍水明', '杨宝圣', '肖健', '谭泽松'],
            '软件部': ['郭文彬', '李陇杰', '孔德枞', '罗嘉俊', '屈中翔', '阮云泉', '陶松', '阳俊'],
            '开发部': ['林秋宇', '欧佰康', '孙彪', '游凯文', '张晓光', '周思捷'],
            '总经办-技术部': ['章魏康', '钟远金', '唐硕', '陈广', '赵明昊', '刘俊浩', '何博轩', '陈楷锐', '郭婉婷', '郭总', '黄永朗', 
                        '黄梓聪', '赖晓梅', '罗威', '庞优华', '王畅', '张湘子', '张紫荷', '郑妮斯', '黄慧仪', ]
        }
        data = data[data['date'] == date]

        record = data.loc[data['Name'].isin(department_dict['数据部'])]
        invest = data.loc[data['Name'].isin(department_dict['投资部'])]
        software = data.loc[data['Name'].isin(department_dict['软件部'])]
        develop = data.loc[data['Name'].isin(department_dict['开发部'])]
        tech = data.loc[data['Name'].isin(department_dict['总经办-技术部'])]
        total_list = [record, invest, software, develop, tech]
        department_list = ['record', 'invest', 'software', 'develop', 'tech']
        sum_list = []
        for i in range(0, len(total_list)):
            temp_dict = {}
            df = total_list[i][['Name', '综合', 'IM-日内多', 'IM-日内空']]
            df.rename(columns={'Name': 'name', '综合': 'compre', 'IM-日内多': 'dayLong', 'IM-日内空': 'dayShort'}, inplace=True)
            df['compre'] = df['compre'] * 2
            df_dict = df.to_dict(orient='records')
            temp_dict['department'] = department_list[i]
            temp_dict['data'] = df_dict
            sum_list.append(temp_dict)
        

        return_dict['result'] = sum_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
    


## IM走势图
@app.route('/day_in/IM_trend', methods=['GET'])
# @cache.memoize(timeout=3600)
def IM_trend():
    """
    功能：
        返回IM走势图

    接口访问示例：
    http://192.168.10.237:8881/day_in/IM_trend?date=2024-08-05
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取参数
        get_data_pramers = request.args.to_dict()
        date = get_data_pramers.get('date')
        detail = ths.history_future(symbol='IM', start_time=date, end_time=date, period='1m', df=True)[['eob', 'close']]
        detail.rename(columns={'eob': 'timestamp'}, inplace=True)
        day = pro.fut_daily(ts_code='IM.CFX', start_date=transfer_date(date), end_date=transfer_date(date)).sort_values('trade_date', ascending=True).reset_index(drop=True)
        change = day['change1'].iloc[0]
        change_ratio = change / day['pre_close'].iloc[0] * 100
        swing = day['high'].iloc[0] - day['low'].iloc[0]
        swing_ratio = swing / day['pre_close'].iloc[0] * 100
        date_change_list = []
        for time in detail['timestamp'].tolist():
            date_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")  
            timestamp = date_obj.timestamp()  
            date_change_list.append(timestamp)

        date = datetime.strptime(date, "%Y-%m-%d")  
        date = date_obj.timestamp()  
        detail['timestamp'] = date_change_list

        total_dict = {'date': date, 'change': change, 'changeRatio': change_ratio, 'swing': swing, 'swingRatio': swing_ratio, 'detail': detail.to_dict(orient='records')}
        return_dict['result'] = total_dict
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)




##################################################### 其他 ##################################################

## 趋势日数据
@app.route('/day_in/Trend_data', methods=['GET'])
# @cache.memoize(timeout=3600)
def Trend_data():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Trend_data
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
                return_dict['return_code'] = '504'
                return_dict['return_info'] = '请求参数为空'
                return json.dumps(return_dict, ensure_ascii=False)
        
        # 获取传入的参数
        data = Z_Read.trend_data()
        data = pd.DataFrame(data)
        date_list = data['trade_date'].unique().tolist()
        data = data[['return', 'amplitude', 'period']]
        df_list = []
        for i in range(0, len(date_list)):
            temp = data.iloc[i]
            single_result = {}
            single_result['date'] = date_list[i]
            single_result['data'] = temp.to_dict()
            df_list.append(single_result)
        return_dict['result'] = df_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)
    

# 所有参赛人
@app.route('/day_in/All_participant', methods=['GET'])
# @cache.memoize(timeout=3600)
def All_participant():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/All_participant
    """
# 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
                return_dict['return_code'] = '504'
                return_dict['return_info'] = '请求参数为空'
                return json.dumps(return_dict, ensure_ascii=False)
        
        period_list = []
        data = Z_Read.fetch_player_data2()
        data = pd.DataFrame(data)
        grouped = data.groupby('name')
        for name, temp in grouped:
            single_result = {}
            try:
                period = temp['期数'].tolist()
            except:
                period = []
            single_result['name'] = name
            single_result['period'] = period
            period_list.append(single_result)
        return_dict['result'] = period_list
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)



## 综合表现
@app.route('/day_in/Compre_presentation', methods=['GET'])
# @cache.memoize(timeout=3600)
def Compre_presentation():
    """
    功能：
        返回个人指标


    接口访问示例：
    http://192.168.10.237:8881/day_in/Compre_presentation
    """

        # 默认返回内容
    return_dict = {'code': 200, 'result': False, 'msg': '请求成功'}
    try:
        # 判断入参是否为空
        if request.args is None:
            return_dict['return_code'] = '504'
            return_dict['return_info'] = '请求参数为空'
            return json.dumps(return_dict, ensure_ascii=False)
        date_change_list = []
        df = pd.DataFrame()
        total_list = []
        data = Z_Read.fetch_daily_data()
        data = pd.DataFrame(data)
        
        date_list = data['date'].unique().tolist()
        reference = pro.fut_daily(ts_code='IM.CFX', start_date=transfer_date(date_list[0]), end_date=transfer_date(date_list[-1])).sort_values('trade_date', ascending=True)
        # reference = ths.history_future(symbol='IM', start_time='2024-05-10', end_time=date_list[-1], period='1d', df=True)[['eob', 'close']]
        # 当日期货涨幅
        reference['change'] = (reference['change1'] / reference['pre_close']) * 100
        reference = reference[['change']]
        reference = (1+reference['change']*0.01).cumprod()
        reference = (reference-1)*100
        reference = reference.reset_index(drop=True)
        
        
        grouped = data.groupby('date')
        df_list = []
        for date, temp in grouped:
            temp_1 = temp[['综合', 'IM-日内多', 'IM-日内空']]
            temp_1 = temp_1.mean()
            temp_1 = pd.DataFrame(temp_1).T
            df_list.append(temp_1)
        df_total = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_total = (1+df_total*0.01).cumprod()
        df_total = (df_total-1)*100

        df_total = df_total.reset_index(drop=True)
        df_total = pd.concat([df_total, reference], axis=1)
        df_total.rename(columns={'综合': 'compreReturn', 'IM-日内多': 'dayLong', 'IM-日内空': 'dayShort'}, inplace=True)
        df_total['compreReturn'] = df_total['compreReturn'] * 2
        # df_total.to_excel('df_total.xlsx', index=False)
        premium = df_total.iloc[-1]['compreReturn'] - df_total.iloc[-1]['change']
        start_list, end_list = get_match()
        period = len(start_list)-1

        grouped = data.groupby('Name')
        for name, temp in grouped:
            df = temp[['综合', 'IM-日内多', 'IM-日内空']]
            df.rename(columns={'综合': 'compre', 'IM-日内多': 'dayLong', 'IM-日内空': 'dayShort'}, inplace=True)
            
            df = df.sum()
            df = pd.DataFrame(df).T
            df['name'] = name

            df_list.append(df)
        total_df = pd.concat(df_list)
        total_df.sort_values('compre', ascending=False, inplace=True)
        one = total_df.iloc[0]['name']
        
        

        result = {'dayName': one, 'dayPeriod': period, 'dayExceed': premium, 
                    'fiveName': None, 'fivePeriod': None, 'fiveExceed': None,
    }

        
        return_dict['result'] = Z_Read.NanEncoder()._nan_to_none(result)
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)   

    except:
        return_dict['code'] = 400
        return_dict['msg'] = '请求失败'
        return json.dumps(return_dict, ensure_ascii=False, cls=Z_Read.DateEncoder)




if __name__ == "__main__":
    # 启动Flask服务，指定主机IP和端口
    app.run(host='0.0.0.0', port=8881)