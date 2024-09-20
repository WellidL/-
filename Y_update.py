########################################################## 每日数据更新 ############################################################
import pandas as pd
import new_data_source as ds
import datetime
from sqlalchemy import create_engine
import pymysql
import schedule
import time

tacitic = ds.TacticRace('dzkj002','djct666666')
df = tacitic.query_project()


def get_data(participant_id, project_id):
    data = pd.DataFrame()
    # 获取数据信息
    
    # 获取项目名称
    project_data = tacitic.query_project()
    topic_data = tacitic.query_topic(project_id)
    participant_data = tacitic.query_participant(project_id)

    # 获取项目名称
    project_data_dict = project_data.set_index('projectId')['projectName'].to_dict()
    project_name = project_data_dict[project_id]
    # 获取题目信息
    topic_id_dict = topic_data.set_index('topicId').to_dict('index')

    # 获取参赛者信息
    participant_id_dict = participant_data.set_index('participantId')['userName'].to_dict()
    data_list = []
    for each_topic_id in topic_id_dict:
        topic_name = topic_id_dict[each_topic_id]['topicName']
        topic_start_date = topic_id_dict[each_topic_id]['start_date']
        topic_end_date = topic_id_dict[each_topic_id]['end_date']
        signal_data_dict = tacitic.query_single_paricipant_signal(each_topic_id,participant_id,topic_start_date,topic_end_date)
        if not signal_data_dict:
            continue
                
        data = pd.DataFrame(signal_data_dict[topic_name]['transactionList'])
        data['dealDate'] = data['dealDate'].apply(lambda x:datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
        rename_columns = {
            'dealDate':'委托时间',
            'security':'品种',
            'stock':'标的',
            'transaction':'交易类型',
            'type':'下单类型',
            'orderAmount':'委托数量',
            'price':'成交价',
            'turnover':'成交额',
            'trueOrderAmount':'成交数量',
            'status':'状态',
            'gains':'平仓盈亏',
            'commission':'手续费'
        }
        data = data.rename(columns=rename_columns)
        data = data[list(rename_columns.values())]
        data['project'] = topic_name
        data_list.append(data)
    try:
        data_total = pd.concat(data_list, axis=0)
        return data_total
    except:
        return None



def connect_sql():
    """
    功能：连接mysql数据库
    """
    connection = pymysql.connect( 
        host='192.168.10.210',  # 数据库主机名,3号虚拟机
        port=3306,               # 数据库端口号，默认为3306
        user='root',             # 数据库用户名
        passwd='djct003',         # 数据库密码
        db='dayin',               # 数据库名称
        charset='utf8'           # 字符编码
    )
    return connection


database_dayin = 'mysql+pymysql://root:djct003@192.168.10.210/dayin'
database_five_day = 'mysql+pymysql://root:djct003@192.168.10.210/five_day'



def day_update():  
    # 假设df是已经加载的DataFrame  
    filtered_df = df[df['projectName'].str.contains('(日内)')]  
    projectId_list = filtered_df['projectId'].tolist()[-2:]  # 只取最后两个项目ID  
  
    project_data = tacitic.query_project()  # 确保这个函数存在且正常工作  
    data_list = []  
  
    for projectId in projectId_list:  
        topic_data = tacitic.query_topic(projectId)  # 确保这个函数返回所需数据  
        participant_data = tacitic.query_participant(projectId)  
  
        participant_id_dict = participant_data.set_index('participantId')['userName'].to_dict()  
  
        all_dict = {}  
        for participant_id, participant_name in participant_id_dict.items():  
            data = get_data(participant_id, projectId)  # 确保这个函数返回一个DataFrame  
            all_dict[participant_name] = data  
  
        data_list.append(all_dict)  
  
    result_dict = {}  
    for dict1, dict2 in zip(data_list, data_list[1:]):  # 使用zip来遍历相邻的两个字典  
        combined_dict = {}  
        for key in dict1.keys() | dict2.keys():  
            df1, df2 = dict1.get(key, None), dict2.get(key, None)  
            if df1 is not None and df2 is not None:  
                combined_dict[key] = pd.concat([df1, df2], axis=0, ignore_index=True)  
            elif df1 is not None:  
                combined_dict[key] = df1  
            elif df2 is not None:  
                combined_dict[key] = df2  
        result_dict.update(combined_dict)  
  
    engine = create_engine(database_dayin)  
    for participant, data in result_dict.items():  
        if data.empty:  
            continue  
  
        table_name = participant  # 简单的表名清理，避免SQL保留字和特殊字符  
        table_name = f"`{table_name}`"  # 使用反引号包围表名  
  
        try:  
            query = f"SELECT * FROM {table_name}"  
            df_db = pd.read_sql(query, con=engine)  
  
            if df_db.empty:  
                data.to_sql(participant, con=engine, if_exists='append', index=False)  
            else:  
                # 假设'委托时间'和'project'是存在的列  
                df_db_indexed = df_db.set_index(['委托时间', 'project'])  
                data_indexed = data.set_index(['委托时间', 'project'])  
                to_add = data_indexed[~data_indexed.index.isin(df_db_indexed.index)]  
                if not to_add.empty:  
                    to_add.reset_index().to_sql(participant, con=engine, if_exists='append', index=False)  
  
        except Exception as e:  
            print(f"Error processing data for {participant}: {e}")





##################################################### 个人每期数据 #####################################################
import new_data_source as ds
import datetime
from sqlalchemy import create_engine
import pymysql
import schedule
import time
import pandas as pd

tacitic = ds.TacticRace('dzkj002','djct666666')
df = tacitic.query_project()


def get_data(participant_id, project_id):
    data = pd.DataFrame()
    # 获取数据信息
    
    # 获取项目名称
    project_data = tacitic.query_project()
    topic_data = tacitic.query_topic(project_id)
    participant_data = tacitic.query_participant(project_id)

    # 获取项目名称
    project_data_dict = project_data.set_index('projectId')['projectName'].to_dict()
    project_name = project_data_dict[project_id]
    # 获取题目信息
    topic_id_dict = topic_data.set_index('topicId').to_dict('index')

    # 获取参赛者信息
    participant_id_dict = participant_data.set_index('participantId')['userName'].to_dict()
    data_list = []
    for each_topic_id in topic_id_dict:
        topic_name = topic_id_dict[each_topic_id]['topicName']
        topic_start_date = topic_id_dict[each_topic_id]['start_date']
        topic_end_date = topic_id_dict[each_topic_id]['end_date']
        signal_data_dict = tacitic.query_single_paricipant_signal(each_topic_id,participant_id,topic_start_date,topic_end_date)
        if not signal_data_dict:
            continue
                
        data = pd.DataFrame(signal_data_dict[topic_name]['transactionList'])
        data['dealDate'] = data['dealDate'].apply(lambda x:datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
        rename_columns = {
            'dealDate':'委托时间',
            'security':'品种',
            'stock':'标的',
            'transaction':'交易类型',
            'type':'下单类型',
            'orderAmount':'委托数量',
            'price':'成交价',
            'turnover':'成交额',
            'trueOrderAmount':'成交数量',
            'status':'状态',
            'gains':'平仓盈亏',
            'commission':'手续费'
        }
        data = data.rename(columns=rename_columns)
        data = data[list(rename_columns.values())]
        data['project'] = topic_name
        data_list.append(data)
    try:
        data_total = pd.concat(data_list, axis=0)
        return data_total
    except:
        return None
    
def connect_sql():
    """
    功能：连接mysql数据库
    """
    connection = pymysql.connect( 
        host='192.168.10.210',  # 数据库主机名,3号虚拟机
        port=3306,               # 数据库端口号，默认为3306
        user='root',             # 数据库用户名
        passwd='djct003',         # 数据库密码
        db='tactic_data',               # 数据库名称
        charset='utf8'           # 字符编码
    )
    return connection


database_dayin = 'mysql+pymysql://root:djct003@192.168.10.210/tactic_data'


def all_day_update():  
    # 假设df是已经加载的DataFrame  
    filtered_df = df[df['projectName'].str.contains('(日内)')]  
    projectId_list = filtered_df['projectId'].tolist()[-2:]  # 只取最后两个项目ID  
  
    project_data = tacitic.query_project()  # 确保这个函数存在且正常工作  
    all_data_df = pd.DataFrame()  # 初始化一个空的DataFrame来存储所有参与者的数据  
    data_list = []
    for projectId in projectId_list:  
        topic_data = tacitic.query_topic(projectId)  # 确保这个函数返回所需数据  
        participant_data = tacitic.query_participant(projectId)  
  
        participant_id_dict = participant_data.set_index('participantId')['userName'].to_dict()  
        project_index = projectId_list.index(projectId)
        for participant_id, participant_name in participant_id_dict.items():  
            # try:
            data = get_data(participant_id, projectId)  # 确保这个函数返回一个DataFrame  
            if data is not None:  
                # 添加参与者名称到DataFrame中  
                data['name'] = participant_name  
                data['period'] = project_index + 1
                # all_data_df = pd.concat([all_data_df, data], axis=0, ignore_index=True)  
                data_list.append(data)  
                
            else:
                continue
    all_data_df = pd.concat(data_list, axis=0, ignore_index=True)
    # 创建数据库引擎  
    engine = create_engine(database_dayin)  
  
    # 检查数据库中已存在的数据  
    try:  
        query = "SELECT * FROM all_data"  
        df_db = pd.read_sql(query, con=engine)  
  
        # 合并现有数据和要更新的数据，同时去重  
        # 假设'委托时间'、'project'和'name'是判断重复的列  
        all_data_df_merged = pd.concat([df_db, all_data_df], axis=0, ignore_index=True)  
        all_data_df_merged.drop_duplicates(subset=['委托时间', 'project', 'name', 'period'], inplace=True, keep='first')  
  
        # 如果合并后的数据比原数据库数据多，则更新数据库  
        if len(all_data_df_merged) > len(df_db):  
            all_data_df_merged.to_sql('all_data', con=engine, if_exists='replace', index=False)  
            print("Database updated successfully!")  
  
    except Exception as e:  
        print(f"Error updating database: {e}")  




################################################## 其他数据更新 ###################################################
# 日内数据
import Z_Update_Process as ZU
import pandas as pd
temp = ZU.Tactic()
all_member_data = temp.single_radar_test()
import pymysql
from sqlalchemy import create_engine

def connect_sql():
    """
    功能：连接mysql数据库
    """
    connection = pymysql.connect(
        host='192.168.10.210',  # 数据库主机名,3号虚拟机
        port=3306,               # 数据库端口号，默认为3306
        user='root',             # 数据库用户名
        passwd='djct003',         # 数据库密码
        db='tactic_data',               # 数据库名称
        charset='utf8'           # 字符编码
    )
    return connection

connection = connect_sql()
database_url = create_engine('mysql+pymysql://root:djct003@192.168.10.210/tactic_data')
df_list = []
for name, df in all_member_data.items():
    df.reset_index(inplace=True)  
    df.rename(columns={'index': '期数'}, inplace=True)
    df['name'] = name
    df_list.append(df)
total_df = pd.concat(df_list)
total_df.dropna(inplace=True)
try:
    with connection.cursor() as cursor:
        total_df.reset_index(inplace=True, drop=True)
        total_df.rename(columns={'index': '期数'}, inplace=True)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `period_person_data` (   
        `期数` INT(10),
        `综合` FLOAT,  
        `进攻` FLOAT,  
        `防守` FLOAT,  
        `盈亏比` FLOAT,  
        `胜率` FLOAT,  
        `稳定性` FLOAT,  
        `信号数量` FLOAT,  
        `综合排名` FLOAT,
        `name` VARCHAR(256)
    );  
    """  
            
        # 执行SQL语句  
        cursor.execute(create_table_sql)  
        total_df.to_sql('period_person_data', database_url, if_exists='replace', index=False)  
              
            
  
    # 提交事务  
    connection.commit()  
  
finally:  
    # 关闭数据库连接  
    connection.close()


######################################### 个人每日详细数据 ####################################
import Z_Update_Process as ZU
personal_data = ZU.Tactic().day_rank()
import pymysql
from sqlalchemy import create_engine
def connect_sql():
    """
    功能: 连接mysql数据库
    """
    connection = pymysql.connect(
        host='192.168.10.210',  # 数据库主机名,3号虚拟机
        port=3306,               # 数据库端口号，默认为3306
        user='root',             # 数据库用户名
        passwd='djct003',         # 数据库密码
        db='tactic_data',               # 数据库名称
        charset='utf8'           # 字符编码
    )
    return connection

connection = connect_sql()
database_url = create_engine('mysql+pymysql://root:djct003@192.168.10.210/tactic_data')
try:
    with connection.cursor() as cursor:
        
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `primary_data` (   
        `Name` VARCHAR(255),
        `综合` FLOAT,  
        `IM-日内多` FLOAT,  
        `IM-日内空` FLOAT,  
        `date` Char(10) 
    );  
    """  
              
            # 执行SQL语句  
        cursor.execute(create_table_sql)  
        personal_data.to_sql('primary_data', database_url, if_exists='replace', index=False)  
            
            
  
    # 提交事务  
    connection.commit()  
  
finally:  
    # 关闭数据库连接  
    connection.close()



################################################ 趋势日数据 ################################################
import Z_Update_Process as ZU
trend_data = ZU.Tactic().trend_data()
import pymysql
from sqlalchemy import create_engine
def connect_sql():
    """
    功能: 连接mysql数据库
    """
    connection = pymysql.connect(
        host='192.168.10.210',  # 数据库主机名,3号虚拟机
        port=3306,               # 数据库端口号，默认为3306
        user='root',             # 数据库用户名
        passwd='djct003',         # 数据库密码
        db='tactic_data',               # 数据库名称
        charset='utf8'           # 字符编码
    )
    return connection

connection = connect_sql()
database_url = create_engine('mysql+pymysql://root:djct003@192.168.10.210/tactic_data')
try:
    with connection.cursor() as cursor:
        
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `trend_data` (   
        `trade_date` VARCHAR(255),
        `return` FLOAT,  
        `amplitude` FLOAT,  
        `period` INT 
    );  
    """  
              
            # 执行SQL语句  
        cursor.execute(create_table_sql)  
        trend_data.to_sql('trend_data', database_url, if_exists='replace', index=False)  
            
            
  
    # 提交事务  
    connection.commit()  
  
finally:  
    # 关闭数据库连接  
    connection.close()



################################################ 日内日期数据 ################################################
import pandas as pd
import new_data_source as ds
import pymysql
from sqlalchemy import create_engine

def get_match():
    """
    获得比赛基础数据：比赛时间
    """
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
    return start_list, end_list


start_list, end_list = get_match()
df = pd.DataFrame({
    'start_date': start_list,
    'end_date': end_list
})



def connect_sql():
    """
    功能: 连接mysql数据库
    """
    connection = pymysql.connect(
        host='192.168.10.210',  # 数据库主机名,3号虚拟机
        port=3306,               # 数据库端口号，默认为3306
        user='root',             # 数据库用户名
        passwd='djct003',         # 数据库密码
        db='tactic_data',               # 数据库名称
        charset='utf8'           # 字符编码
    )
    return connection

connection = connect_sql()
database_url = create_engine('mysql+pymysql://root:djct003@192.168.10.210/tactic_data')
try:
    with connection.cursor() as cursor:
        
        
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `date_list` (   
        `start_date` VARCHAR(255),
        `end_date` VARCHAR(255)
    );  
    """  
              
            # 执行SQL语句  
        cursor.execute(create_table_sql)  
        df.to_sql('date_list', database_url, if_exists='replace', index=False)  
            
            
  
    # 提交事务  
    connection.commit()  
  
finally:  
    # 关闭数据库连接  
    connection.close()




###################################### 更新 ######################################
day_update()  
all_day_update()










