import requests
import requests
import pandas as pd
import datetime
import re
from functools import lru_cache
from DjData import ths

##### 登录股指比赛平台 #####
class TacticRace():
    
    def __init__(self,username,password) -> None:
        self.token = self.imitate_login_tactic_competition(username,password)
        # 项目的字典
        self.all_race_dict = self.get_all_race_dict()
        # 数据集合,被调用和加载过的数据皆存储在这里
        self.datas = {}
    
    def get_all_race_dict(self):
        project_dict = self.query_project().set_index('projectId').to_dict('index')
        for project in project_dict:
            topic_dict = self.query_topic(project).set_index('topicId').to_dict('index')
            participant_dict = self.query_participant(project).set_index('participantId').to_dict('index')
            topic_dict['participant_dict'] = participant_dict
            project_dict[project]['topic'] = topic_dict
        return project_dict

    # def write_data(func):
    #     def wrapper(self,*args,**kwargs):
    #         if args:
    #             raise TypeError('只允许使用关键字参数')
    #         result = func(self,*args,**kwargs)
    #         key = (kwargs['topic_id'],kwargs['participant_id'])
    #         if key not in self.datas:
    #             self.datas[key] = result
    #         return result
    #     return wrapper
    
    def imitate_login_tactic_competition(self,username,password):
        """股指策略平台模拟登录

        Args:
            username (str): 用户名
            password (str): 密码

        Returns:
            str: 登录的token
        """
        url = 'http://test46.szdjct.com/competition/login'
        post_data = {
            'username': username, 
            'password': password
            }
        token = requests.post(url,json=post_data).json()['data']['token']
        return token

    def query_project(self):
        """查询项目

        Returns:
            pd.Dataframe: 项目的信息
        """
        url = 'http://test46.szdjct.com/competition/project/list'
        params = {'projectType':2}
        headers = {
            'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Token':self.token
        }
        res = requests.get(url,params=params,headers=headers).json()['data']
        data = pd.DataFrame(res)
        data = data[['projectId','projectName','projectType','uploadState','status']]
        data = data.rename(columns={'projectType':'比赛模式','uploadState':'上传通道'})
        # 比赛模式，1为自由模式，2为推荐策略模式
        data['比赛模式'] = data['比赛模式'].apply(lambda x:'推荐模式' if x == '2' else '自由模式')
        data['上传通道'] = data['上传通道'].apply(lambda x:'关闭' if x == '2' else '开启')
        return data

    def query_topic(self,projectId):
        """查询每个项目中的题目

        Args:
            projectId (int): 项目ID

        Returns:
            返回每个项目的题目相关信息
        """
        url = 'http://test46.szdjct.com/competition/topic/list'
        params = {
            'projectId':projectId,
            'roundId':-1,
            'type':1
            }
        headers = {
            'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Token':self.token
        }
        res = requests.get(url,params=params,headers=headers).json()['data']
        data = pd.DataFrame(res)
        data = data[['projectId','topicId','topicName','initialFunding','minSignalNumber','maxSignalNumber','periodicBandList']]
        data['upload_num'] = data['periodicBandList'].apply(lambda x:x[0]['count'])
        data['start_date'] = data['periodicBandList'].apply(lambda x:x[0]['periodicBand'].split('~')[0])
        data['end_date'] = data['periodicBandList'].apply(lambda x:x[0]['periodicBand'].split('~')[1])
        data = data.rename(columns={'initialFunding':'初始资金','minSignalNumber':'最小总信号数','maxSignalNumber':'最大总信号数'})
        data = data.drop('periodicBandList',axis=1)
        return data

    def query_participant(self,projectId):
        """查询每个项目的参赛者

        Args:
            projectId (int): 项目ID

        Returns:
            返回参赛者信息
        """
        url = 'http://test46.szdjct.com/competition/participant/list'
        params = {
            'projectId':projectId,
            }
        headers = {
            'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Token':self.token
        }
        res = requests.get(url,params=params,headers=headers).json()['data']
        data = pd.DataFrame(res)
        data = data[['projectId','participantId','userId','userName']]
        return data

    # @write_data
    def query_single_paricipant_signal(self,topic_id,participant_id,start_date,end_date):
        """查询参赛者策略信号和收益

        Args:
            topic_id (str|list): 题目id
            participant_id (str|int): 参赛者id
            start_date (str): 开始日期
            end_date (str): 结束日期

        Returns:
            dict: key为题目,value为信息的字典
        """
        if isinstance(topic_id,list):
            # 将列表中int类型转换为str
            topic_id = [str(i) for i in topic_id]
            topic_id = ','.join(topic_id)
        
        url = 'http://test46.szdjct.com/competition/findTransactionList'
        params = {
            'topicIdList':topic_id,
            'participantId':participant_id,
            'startDate':start_date,
            'endDate':end_date
            }
        headers = {
            'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Token':self.token
        }
        res = requests.get(url,params=params,headers=headers).json()['data']
        if res:
            data = pd.DataFrame(res)
            data['name'].iloc[0] = data['name'].iloc[0]['topicName']
            return data.set_index('name').to_dict('index')
        else:
            return None


# if __name__ == "__main__":
#     # username,password = 'djct220108','djct220108'
#     # data = recommend_stock_data_all(username,password)
#     # print(data)
    
#     topic_id = 44
#     participant_id = 60
#     start_date = '2024-06-03'
#     end_date = '2024-06-07'
#     # 
#     tactic = TacticRace('dzkj002','djct666666')
    
#     # topic = tactic.query_topic(65)
#     # participant = tactic.query_participant(65)
#     data = tactic.query_single_paricipant_signal(topic_id=179,participant_id=945,start_date=start_date,end_date=end_date)
#     # print(data)
#     # data = tactic.query_single_paricipant_signal(topic_id=185,participant_id=980,start_date=start_date,end_date=end_date)
#     # data = tactic.query_single_paricipant_signal(topic_id=186,participant_id=980,start_date=start_date,end_date=end_date)
#     # print(data)
#     # print(tactic.datas)

    
#     # 确定项目
#     project_id = 60

#     # 获取数据信息
#     project_data = tactic.query_project()
#     topic_data = tactic.query_topic(project_id)
#     participant_data = tactic.query_participant(project_id)

#     # 获取项目名称
#     project_data_dict = project_data.set_index('projectId')['projectName'].to_dict()
#     project_name = project_data_dict[project_id]
#     # 获取题目信息
#     topic_id_dict = topic_data.set_index('topicId').to_dict('index')
#     # 获取参赛者信息
#     participant_id_dict = participant_data.set_index('participantId')['userName'].to_dict()
    
#     # query_single_paricipant_signal(177,815,'2024-05-17','2024-05-24')
#     # participant_id_dict = {815:'黄慧怡'}

#     for participant_id in participant_id_dict:
#         participant_name = participant_id_dict[participant_id]
#         if_have_excel_writer = False
#         # 先以老版本来生成一份xlsx的文档
#         for each_topic_id in topic_id_dict:
#             topic_name = topic_id_dict[each_topic_id]['topicName']
#             topic_start_date = topic_id_dict[each_topic_id]['start_date']
#             topic_end_date = topic_id_dict[each_topic_id]['end_date']
#             signal_data_dict = tactic.query_single_paricipant_signal(each_topic_id,participant_id,topic_start_date,topic_end_date)
#             if not signal_data_dict:
#                 continue
#             else:
#                 if not if_have_excel_writer:
#                     excel_path = f'5日策略信号/{project_name}-{participant_name}.xlsx'
#                     excel_writer = pd.ExcelWriter(excel_path)
#                     if_have_excel_writer = True
#             print(signal_data_dict)
#             data = pd.DataFrame(signal_data_dict[topic_name]['transactionList'])
#             data['dealDate'] = data['dealDate'].apply(lambda x:datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S'))
#             rename_columns = {
#                 'dealDate':'委托时间',
#                 'security':'品种',
#                 'stock':'标的',
#                 'transaction':'交易类型',
#                 'type':'下单类型',
#                 'orderAmount':'委托数量',
#                 'price':'成交价',
#                 'turnover':'成交额',
#                 'trueOrderAmount':'成交数量',
#                 'status':'状态',
#                 'gains':'平仓盈亏',
#                 'commission':'手续费'
#             }
#             data = data.rename(columns=rename_columns)
#             data = data[list(rename_columns.values())]
#             # 写入excel中一个表
#             data.to_excel(excel_writer,sheet_name=topic_name,index=False)
        
#         try:
#             excel_writer.close()
#         except: 
#             print(f'{participant_name}未参加本次比赛')
            
            