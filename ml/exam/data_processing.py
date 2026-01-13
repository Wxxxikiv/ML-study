import datetime
import pandas as pd
from chinese_calendar import is_workday
# from utils import is_workday

base_time = "20191227220000"
base_date = datetime.datetime.strptime(base_time, "%Y%m%d%H%M%S").date()


def transform_time(time_str):
    time_str = str(time_str).split('.')[0]
    hour = int(str(time_str)[8:])
    period = ""
    day_type = ""
    if 23 <= hour <= 24 or 0 <= hour <= 6:
        period = '3'
    elif 8 <= hour <= 18:
        period = '1'
    elif 19 <= hour <= 22 or hour == 7:
        period = '2'

    time_str1 = str(time_str)+"0000"
    # 先转换为时间数组
    date = datetime.datetime.strptime(time_str1, "%Y%m%d%H%M%S").date()
    timestamp = datetime.datetime.strptime(time_str1, "%Y%m%d%H%M%S").timestamp()
    date_num = (date - base_date).days
    if is_workday(date):
        day_type = '1'
    else:
        day_type = '2'
    return [date_num, hour, period, day_type, timestamp]


if __name__ == '__main__':
    filename = "result_data.csv"
    dataset = pd.read_csv(filename)
    x = dataset['operatetime'].values

    day = []
    hour = []
    period = []
    day_type = []
    timestamp = []
    for time_str in x:
        arr = transform_time(time_str)
        day.append(arr[0])
        hour.append(arr[1])
        period.append(arr[2])
        day_type.append(arr[3])
        timestamp.append(arr[4])
    dataset['day'] = day
    dataset['hour'] = hour
    dataset['period'] = period
    dataset['day_type'] = day_type
    dataset['timestamp'] = timestamp
    dataset.to_csv("final_"+filename)

