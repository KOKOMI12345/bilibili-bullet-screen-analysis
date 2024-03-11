from work_dependent import *


class AsyncGetBiliBiliAPI:
    def __init__(self) -> None:
        self.oid_url = f"https://api.bilibili.com/x/web-interface/view?bvid="
        self.log = LogManager().GetLogger("CommentThread")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.12",
            "Refener": "https://www.bilibili.com"
        }

    async def Translate_bv_to_oid(self,bv:str) -> int:
        self.log.info("开始转换BV号为OID")
        async with httpx.AsyncClient(max_redirects=3,timeout=5) as client:
            try:
                response = await client.get(self.oid_url+bv,headers=self.headers)
                data = response.json()
                if data['code'] == 0:
                    oid = data['data']['cid']
                    self.log.info(f"转换成功，BV号：{bv}，OID：{oid}")
                    return oid
                else:
                    self.log.warning(f"转换失败，BV号：{bv}，错误信息：{data['message']}")
                    return None
            except Exception as e:
                self.log.warning(f"转换失败，BV号：{bv}，错误信息：{e}")
                return None

    async def get_comment(self,oid:int,max_retry:int = 3,timeout:float = 5.0) -> str | None:
        async with httpx.AsyncClient(timeout=timeout,max_redirects=max_retry) as client:
            try:
                self.log.info("开始获取评论")
                url = f"https://api.bilibili.com/x/v1/dm/list.so?oid={oid}"
                response = await client.get(url=url,headers=self.headers)
                self.log.debug(f"获取评论状态码：{response.status_code}")
                if response.status_code == 200:
                    pass
                elif response.status_code == 404:
                    self.log.warning("该弹幕或评论未存在或已经关闭")
                self.log.info("获取评论成功")
                return response.text
            except Exception as e:
                stack_info = traceback.format_exc()
                self.log.warning(f"获取评论失败，错误信息：{stack_info}")

    async def Paser_XML(self,xml_data:str) -> tuple[list,int]:
        self.log.info("开始解析XML")
        # 返回包含弹幕的列表和弹幕的条数
        try:
            root = ET.fromstring(xml_data)
            data_cache = [d.text for d in root.findall('d')]
            lengith = len(data_cache)
            self.log.info("解析XML成功")
            return data_cache , lengith
        except Exception as e:
            stack_info = traceback.format_exc()
            self.log.warning(f"解析XML失败，错误信息：{stack_info}")
    
    async def GetComments(self,bv:str) -> dict[int,list[str]]:
        try:
            self.log.info("开始获取弹幕")
            oid = await self.Translate_bv_to_oid(bv=bv)
            comment_data = await self.get_comment(oid=oid)
            parsed_data, length = await self.Paser_XML(comment_data)
            self.log.info(f"弹幕数量：{length}")
            self.log.info("获取弹幕成功")
            dat = {"len":length,"data":parsed_data}
            return dat
        except Exception as e:
            stack_info = traceback.format_exc()
            self.log.critical(f"获取弹幕失败，错误信息：{e}\n{stack_info}")
            return {"len":0,"data":[]}

if __name__ == '__main__':
   import asyncio

async def standardization_data(data:dict):
    model = {
        "code": 200,
        "data" : {
            "len": data['len'],
            "comments": data['data']
        },
        "msg": "OK"
    }
    opt_js = json.dumps(model,ensure_ascii=False,indent=4)
    return opt_js

async def main():
    api = AsyncGetBiliBiliAPI()
    data = await api.GetComments("BV1hy421i7ji")
    jsons = await standardization_data(data)
    with open("data.json","w",encoding="utf-8") as f:
        f.write(jsons)
    api.log.info("数据标准化完成")



loop = asyncio.get_event_loop()
loop.run_until_complete(main())
