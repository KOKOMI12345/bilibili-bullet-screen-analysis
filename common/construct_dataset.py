import asyncio
import json
import csv

from get_data import AsyncGetBiliBiliAPI
from get_data import standardization_data
async def main():
    api = AsyncGetBiliBiliAPI()
    # TODO: 循环bv,如何爬取多个视频？
    data = await api.GetComments("BV1hy421i7ji")
    json_data = await standardization_data(data)

    # 保存json
    with open("data_set.json", "w", encoding="utf-8") as f:
        f.write(json_data)
    api.log.info("数据标准化完成")

    # 读取JSON文件
    with open('data_set.json', 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # 提取评论数据
    comments = json_data['data']['comments']

    # 存储为CSV文件
    csv_file = 'data_from_json.csv'
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])
        for text in comments:
            writer.writerow([text])

    print("数据已成功保存为CSV文件")

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
