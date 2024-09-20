import pymysql
import datetime
import json
import pandas as pd
from decimal import Decimal
from datetime import date
from sqlalchemy import create_engine




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



######################## 读取个人数据 ##########################
## 个人每期数据
def fetch_player_data(name):  
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


## 个人每期数据2
def fetch_player_data2():  
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
                    sql_query = f"SELECT * FROM `period_person_data`"  
                    
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



## 个人每日数据
def fetch_daily_data():  
        """  
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
                    sql_query = f"SELECT * FROM `primary_data`"  
                    
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







## 趋势数据
def trend_data():
    """  
        趋势日数据
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
                sql_query = f"SELECT * FROM `trend_data`"  
                
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



# 日期数据
def date_data():
    """  
        趋势日数据
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
                sql_query = f"SELECT * FROM `date_list`"  
                
                cursor.execute(sql_query)  
                
                # 获取所有记录  
                records = cursor.fetchall()  
                records = pd.DataFrame(records)
                # 如果使用了DictCursor，则records中的每个元素都是字典；否则是元组  
                return records  
    except pymysql.MySQLError as e:  
        print(f"Error: {e}")  
        return None  
    finally:  
        # 无论是否发生异常，都尝试关闭连接  
        connection.close()  


# 所有开平仓数据
def all_data():
    """  
        所有开平仓数据
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
                sql_query = f"SELECT * FROM `all_data`"  
                
                cursor.execute(sql_query)  
                
                # 获取所有记录  
                records = cursor.fetchall()  
                records = pd.DataFrame(records)
                # 如果使用了DictCursor，则records中的每个元素都是字典；否则是元组  
                records = pd.DataFrame(records)
                return records  
    except pymysql.MySQLError as e:  
        print(f"Error: {e}")  
        return None  
    finally:  
        # 无论是否发生异常，都尝试关闭连接  
        connection.close()  


# 所有期数据
def period_data():
    """  
        所有期数据
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
                sql_query = f"SELECT * FROM `new_period_data`"  
                
                cursor.execute(sql_query)  
                
                # 获取所有记录  
                records = cursor.fetchall()  
                records = pd.DataFrame(records)
                # 如果使用了DictCursor，则records中的每个元素都是字典；否则是元组  
                records = pd.DataFrame(records)
                return records  
    except pymysql.MySQLError as e:  
        print(f"Error: {e}")  
        return None  
    finally:  
        # 无论是否发生异常，都尝试关闭连接  
        connection.close()  