import json
import os

# ==========================================
# 超市机器人 - 智能提货/补货 映射与配置系统 (detect_12 版)
# 来源: C:\Users\OMENXCJ\Desktop\detect_12\data.yaml
# ==========================================

YOLO_LABELS = [
    "ADgainai",
    "Opaoguonaicaomei",
    "Opaoguonaiyuanwei",
    "RIOcaomei",
    "RIOjuzi",
    "RIOlizhi",
    "RIOputao",
    "RIOqingti",
    "RIOrusuanjun",
    "RIO_shuimitao",
    "Weitanai",
    "anmuxi",
    "asamudi",
    "asamumoli",
    "asamuyuanwei",
    "Baishi",
    "baiwei",
    "binghongchakang",
    "binglvchakang",
    "chunyue",
    "dizhigaogainai",
    "doleboluo",
    "donggua",
    "dongpengteyin",
    "fenda",
    "gaogainiuru",
    "guangminniunai",
    "hongdouyimishui",
    "hongniu",
    "hongniuao",
    "hongniuhunheshuiguo",
    "hongniuzengqiang",
    "hongzaogouqishui",
    "jiaduobao",
    "jianjiaocheng",
    "jianjiaolan",
    "jianjiaolv",
    "jianjiaozi",
    "jindian",
    "jinniangbaipijiu",
    "jinniangpijiu",
    "jinyinhua",
    "kebikefanqie",
    "kekoukele",
    "kekoukeleoreo",
    "kekoukelewutang",
    "kuangquanshuihekaishui",
    "kuer",
    "maidong",
    "maidongdianjiezhi",
    "mangguozhi",
    "meizhiyuanchenzi",
    "meizhiyuanputao",
    "meizhiyuanqinti",
    "mengniuchunniunai",
    "moliqingchakang",
    "mozhaobaixiangguo",
    "mozhaoputao",
    "mozhaoqiyiguo",
    "mozhaowutang",
    "mozhaoyuanwei",
    "naixueguocha",
    "nimenggongheguoxiangshuiningmeng",
    "ningmenggongheguobale",
    "nongfushanquan",
    "pai tea lemonblacktea",
    "pai tea mitaowulong",
    "pai tea molilemon",
    "pai tea pomelogreentea",
    "qingguocha",
    "qingmeilvcha",
    "quchenshikangpucha",
    "quchenshimojituo",
    "quchenshiwutang",
    "quchenshiyanwei",
    "quchenshiyuanwei",
    "rusuanjun",
    "sandelijupiwulong",
    "sandelimoliwulong",
    "shuangwaiwai",
    "shuidonglemeiguo",
    "shuidongleningmeng",
    "shuidongletaozi",
    "shuirongCchenpi",
    "shuirongCningmen",
    "shuirongCxiyou",
    "shupianhuanggualeshi",
    "shupianleshi",
    "shupianningmengleshi",
    "shupianpinke",
    "shupianshaokaoleshi",
    "shupianshaokaopinke",
    "vitajuhua",
    "vitaningmen",
    "vitapingguo",
    "wahaharusuanjun",
    "wanglaoji",
    "wanglaojihe",
    "Wangzai",
    "wulongchajupiwulong",
    "wulongchamoliwulong",
    "xianguocheng",
    "xinbakenatie",
    "xinbaketaotaowulong",
    "xuehua",
    "xuehuadan",
    "yakult",
    "yezhi",
    "yida",
    "yidamigua",
    "yilichunniunai",
    "yingyangkuaixian",
    "yingyangkuaixian hongzao",
    "yingyangkuaixian yezi",
    "yonghuiniunai",
    "yousuanruadgai",
    "yousuanruboluo",
    "yousuanrucaomei",
    "yousuanrulanmei",
    "yousuanrulizhi",
    "yousuanruxinzi",
    "yousuanruyuanwei",
    "zumingyoujidouru",
]

# ------------------------------------------
# 2. 视觉特征词库字典
# ------------------------------------------
CHINESE_ITEM_DB = {
    "AD钙奶": ["ADgainai"],
    "O泡果奶": ["Opaoguonaicaomei", "Opaoguonaiyuanwei"],
    "O泡果奶草莓味": ["Opaoguonaicaomei"],
    "O泡果奶原味": ["Opaoguonaiyuanwei"],
    "RIO": ["RIOcaomei", "RIOjuzi", "RIOlizhi", "RIOputao", "RIOqingti", "RIOrusuanjun", "RIO_shuimitao"],
    "锐澳": ["RIOcaomei", "RIOjuzi", "RIOlizhi", "RIOputao", "RIOqingti", "RIOrusuanjun", "RIO_shuimitao"],
    "RIO草莓": ["RIOcaomei"],
    "RIO橘子": ["RIOjuzi"],
    "RIO荔枝": ["RIOlizhi"],
    "RIO葡萄": ["RIOputao"],
    "RIO青提": ["RIOqingti"],
    "RIO乳酸菌": ["RIOrusuanjun"],
    "RIO水蜜桃": ["RIO_shuimitao"],
    "维他奶": ["Weitanai", "vitajuhua", "vitaningmen", "vitapingguo"],
    "维他菊花茶": ["vitajuhua"],
    "维他柠檬茶": ["vitaningmen"],
    "维他苹果": ["vitapingguo"],
    "安慕希": ["anmuxi"],
    "阿萨姆": ["asamudi", "asamumoli", "asamuyuanwei"],
    "阿萨姆原味": ["asamuyuanwei"],
    "阿萨姆茉莉": ["asamumoli"],
    "百事可乐": ["Baishi"],
    "百威": ["baiwei"],
    "冰红茶": ["binghongchakang"],
    "冰绿茶": ["binglvchakang"],
    "纯悦": ["chunyue"],
    "低脂高钙奶": ["dizhigaogainai"],
    "高钙奶": ["dizhigaogainai", "gaogainiuru"],
    "都乐菠萝汁": ["doleboluo"],
    "冬瓜茶": ["donggua"],
    "东鹏特饮": ["dongpengteyin"],
    "芬达": ["fenda"],
    "光明牛奶": ["guangminniunai"],
    "红豆薏米水": ["hongdouyimishui"],
    "红牛": ["hongniu", "hongniuao", "hongniuhunheshuiguo", "hongniuzengqiang"],
    "红牛混合水果": ["hongniuhunheshuiguo"],
    "红牛增强": ["hongniuzengqiang"],
    "红枣枸杞水": ["hongzaogouqishui"],
    "加多宝": ["jiaduobao"],
    "尖叫": ["jianjiaocheng", "jianjiaolan", "jianjiaolv", "jianjiaozi"],
    "尖叫橙味": ["jianjiaocheng"],
    "尖叫蓝味": ["jianjiaolan"],
    "尖叫绿味": ["jianjiaolv"],
    "尖叫紫味": ["jianjiaozi"],
    "金典": ["jindian"],
    "金酿白啤酒": ["jinniangbaipijiu"],
    "金酿啤酒": ["jinniangpijiu"],
    "金银花": ["jinyinhua"],
    "可比克薯片": ["kebikefanqie"],
    "可比克番茄味薯片": ["kebikefanqie"],
    "可口可乐": ["kekoukele", "kekoukeleoreo", "kekoukelewutang"],
    "可口可乐奥利奥": ["kekoukeleoreo"],
    "可口可乐无糖": ["kekoukelewutang"],
    "矿泉水": ["chunyue", "kuangquanshuihekaishui", "nongfushanquan"],
    "喝开水": ["kuangquanshuihekaishui"],
    "酷儿": ["kuer"],
    "脉动": ["maidong", "maidongdianjiezhi"],
    "脉动电解质": ["maidongdianjiezhi"],
    "芒果汁": ["mangguozhi"],
    "美汁源": ["meizhiyuanchenzi", "meizhiyuanputao", "meizhiyuanqinti"],
    "美汁源橙汁": ["meizhiyuanchenzi"],
    "美汁源葡萄": ["meizhiyuanputao"],
    "美汁源青提": ["meizhiyuanqinti"],
    "蒙牛纯牛奶": ["mengniuchunniunai"],
    "茉莉清茶": ["moliqingchakang"],
    "魔爪": ["mozhaobaixiangguo", "mozhaoputao", "mozhaoqiyiguo", "mozhaowutang", "mozhaoyuanwei"],
    "魔爪百香果": ["mozhaobaixiangguo"],
    "魔爪葡萄": ["mozhaoputao"],
    "魔爪奇异果": ["mozhaoqiyiguo"],
    "魔爪无糖": ["mozhaowutang"],
    "魔爪原味": ["mozhaoyuanwei"],
    "奈雪果茶": ["naixueguocha"],
    "柠檬共和国": ["nimenggongheguoxiangshuiningmeng", "ningmenggongheguobale"],
    "柠檬共和国香水柠檬": ["nimenggongheguoxiangshuiningmeng"],
    "柠檬共和国芭乐": ["ningmenggongheguobale"],
    "农夫山泉": ["nongfushanquan"],
    "派茶": ["pai tea lemonblacktea", "pai tea mitaowulong", "pai tea molilemon", "pai tea pomelogreentea"],
    "派茶柠檬红茶": ["pai tea lemonblacktea"],
    "派茶蜜桃乌龙": ["pai tea mitaowulong"],
    "派茶茉莉柠檬": ["pai tea molilemon"],
    "派茶西柚绿茶": ["pai tea pomelogreentea"],
    "青果茶": ["qingguocha"],
    "青梅绿茶": ["qingmeilvcha"],
    "屈臣氏": ["quchenshikangpucha", "quchenshimojituo", "quchenshiwutang", "quchenshiyanwei", "quchenshiyuanwei"],
    "屈臣氏康普茶": ["quchenshikangpucha"],
    "屈臣氏莫吉托": ["quchenshimojituo"],
    "屈臣氏无糖": ["quchenshiwutang"],
    "屈臣氏盐味": ["quchenshiyanwei"],
    "屈臣氏原味": ["quchenshiyuanwei"],
    "乳酸菌": ["rusuanjun", "wahaharusuanjun", "yakult"],
    "三得利": ["sandelijupiwulong", "sandelimoliwulong"],
    "三得利橘皮乌龙": ["sandelijupiwulong"],
    "三得利茉莉乌龙": ["sandelimoliwulong"],
    "爽歪歪": ["shuangwaiwai"],
    "水动乐": ["shuidonglemeiguo", "shuidongleningmeng", "shuidongletaozi"],
    "水动乐莓果": ["shuidonglemeiguo"],
    "水动乐柠檬": ["shuidongleningmeng"],
    "水动乐桃子": ["shuidongletaozi"],
    "水溶C": ["shuirongCchenpi", "shuirongCningmen", "shuirongCxiyou"],
    "水溶C陈皮": ["shuirongCchenpi"],
    "水溶C柠檬": ["shuirongCningmen"],
    "水溶C西柚": ["shuirongCxiyou"],
    "薯片": ["kebikefanqie", "shupianhuanggualeshi", "shupianleshi", "shupianningmengleshi", "shupianpinke", "shupianshaokaoleshi", "shupianshaokaopinke"],
    "乐事薯片": ["shupianhuanggualeshi", "shupianleshi", "shupianningmengleshi", "shupianshaokaoleshi"],
    "黄瓜味薯片": ["shupianhuanggualeshi"],
    "青柠味薯片": ["shupianningmengleshi"],
    "原味薯片": ["shupianleshi", "shupianpinke"],
    "品客薯片": ["shupianpinke", "shupianshaokaopinke"],
    "烧烤薯片": ["shupianshaokaoleshi", "shupianshaokaopinke"],
    "娃哈哈乳酸菌": ["wahaharusuanjun"],
    "王老吉": ["wanglaoji", "wanglaojihe"],
    "旺仔牛奶": ["Wangzai"],
    "乌龙茶": ["wulongchajupiwulong", "wulongchamoliwulong", "sandelijupiwulong", "sandelimoliwulong", "xinbaketaotaowulong", "pai tea mitaowulong"],
    "橘皮乌龙": ["wulongchajupiwulong", "sandelijupiwulong"],
    "茉莉乌龙": ["wulongchamoliwulong", "sandelimoliwulong"],
    "鲜橙多": ["xianguocheng"],
    "星巴克拿铁": ["xinbakenatie"],
    "星巴克桃桃乌龙": ["xinbaketaotaowulong"],
    "雪花": ["xuehua", "xuehuadan"],
    "雪花啤酒": ["xuehua", "xuehuadan"],
    "雪花淡爽": ["xuehuadan"],
    "养乐多": ["yakult"],
    "椰汁": ["yezhi"],
    "益达": ["yida", "yidamigua"],
    "益达蜜瓜": ["yidamigua"],
    "伊利纯牛奶": ["yilichunniunai"],
    "营养快线": ["yingyangkuaixian", "yingyangkuaixian hongzao", "yingyangkuaixian yezi"],
    "营养快线红枣": ["yingyangkuaixian hongzao"],
    "营养快线椰子": ["yingyangkuaixian yezi"],
    "永辉牛奶": ["yonghuiniunai"],
    "优酸乳": ["yousuanruadgai", "yousuanruboluo", "yousuanrucaomei", "yousuanrulanmei", "yousuanrulizhi", "yousuanruxinzi", "yousuanruyuanwei"],
    "优酸乳AD钙": ["yousuanruadgai"],
    "优酸乳菠萝味": ["yousuanruboluo"],
    "优酸乳草莓味": ["yousuanrucaomei"],
    "优酸乳蓝莓味": ["yousuanrulanmei"],
    "优酸乳荔枝味": ["yousuanrulizhi"],
    "优酸乳杏子味": ["yousuanruxinzi"],
    "优酸乳原味": ["yousuanruyuanwei"],
    "祖名有机豆乳": ["zumingyoujidouru"],
    "豆乳": ["zumingyoujidouru"],
    "可乐": ["Baishi", "kekoukele", "kekoukeleoreo", "kekoukelewutang"],
    "汽水": ["Baishi", "fenda", "kekoukele", "kekoukeleoreo", "kekoukelewutang"],
    "啤酒": ["baiwei", "jinniangbaipijiu", "jinniangpijiu", "xuehua", "xuehuadan"],
    "纯牛奶": ["guangminniunai", "jindian", "mengniuchunniunai", "yilichunniunai", "yonghuiniunai"],
    "牛奶": ["ADgainai", "anmuxi", "dizhigaogainai", "gaogainiuru", "guangminniunai", "jindian", "mengniuchunniunai", "Wangzai", "yilichunniunai", "yonghuiniunai"],
    "酸奶": ["anmuxi", "yousuanruadgai", "yousuanruboluo", "yousuanrucaomei", "yousuanrulanmei", "yousuanrulizhi", "yousuanruxinzi", "yousuanruyuanwei", "wahaharusuanjun", "yakult"],
    "饮用水": ["chunyue", "kuangquanshuihekaishui", "nongfushanquan"],
    "果汁": ["doleboluo", "kuer", "mangguozhi", "meizhiyuanchenzi", "meizhiyuanputao", "meizhiyuanqinti", "shuirongCchenpi", "shuirongCningmen", "shuirongCxiyou", "vitapingguo", "xianguocheng", "yezhi"],
    "功能饮料": ["dongpengteyin", "hongniu", "hongniuao", "hongniuhunheshuiguo", "hongniuzengqiang", "maidong", "maidongdianjiezhi", "mozhaobaixiangguo", "mozhaoputao", "mozhaoqiyiguo", "mozhaowutang", "mozhaoyuanwei"],
    "茶饮": ["asamudi", "asamumoli", "asamuyuanwei", "binghongchakang", "binglvchakang", "donggua", "jinyinhua", "moliqingchakang", "naixueguocha", "pai tea lemonblacktea", "pai tea mitaowulong", "pai tea molilemon", "pai tea pomelogreentea", "qingguocha", "qingmeilvcha", "sandelijupiwulong", "sandelimoliwulong", "vitajuhua", "vitaningmen", "wanglaoji", "wanglaojihe", "wulongchajupiwulong", "wulongchamoliwulong", "xinbaketaotaowulong"],
}

for _label in YOLO_LABELS:
    CHINESE_ITEM_DB.setdefault(_label, [_label])
    _label_lower = _label.lower()
    if _label_lower != _label:
        CHINESE_ITEM_DB.setdefault(_label_lower, [_label])

# 品牌字典 (强排除过滤：如果明确说了某品牌，剔除拼音里不含该品牌的)
BRAND_SPELLS = {
    "AD钙": ["ADg"],
    "O泡": ["Opaoguonai"],
    "RIO": ["RIO"],
    "锐澳": ["RIO"],
    "维他": ["Weitanai", "vita"],
    "安慕希": ["anmuxi"],
    "阿萨姆": ["asamu"],
    "百事": ["Baishi"],
    "百威": ["baiwei"],
    "纯悦": ["chunyue"],
    "都乐": ["dole"],
    "东鹏": ["dongpeng"],
    "芬达": ["fenda"],
    "光明": ["guangmin"],
    "红牛": ["hongniu"],
    "加多宝": ["jiaduobao"],
    "尖叫": ["jianjiao"],
    "金典": ["jindian"],
    "金酿": ["jinniang"],
    "金银花": ["jinyinhua"],
    "可比克": ["kebike"],
    "可口": ["kekoukele"],
    "酷儿": ["kuer"],
    "脉动": ["maidong"],
    "美汁源": ["meizhiyuan"],
    "蒙牛": ["mengniu"],
    "魔爪": ["mozhao"],
    "奈雪": ["naixue"],
    "农夫山泉": ["nongfushanquan"],
    "派茶": ["pai tea"],
    "屈臣氏": ["quchenshi"],
    "三得利": ["sandeli"],
    "水动乐": ["shuidongle"],
    "水溶C": ["shuirongC"],
    "乐事": ["leshi"],
    "品客": ["pinke"],
    "娃哈哈": ["ADg", "wahaha", "Wangzai"],
    "王老吉": ["wanglaoji"],
    "旺仔": ["Wangzai"],
    "星巴克": ["xinbake"],
    "雪花": ["xuehua"],
    "伊利": ["yili"],
    "益达": ["yida"],
    "优酸乳": ["yousuanru"],
    "永辉": ["yonghui"],
    "祖名": ["zuming"],
}

# 口味字典 (特征过滤：如果包含某个口味，剔除不含对应拼音的)
FLAVOR_SPELLS = {
    "草莓": ["caomei"],
    "原味": ["yuanwei"],
    "葡萄": ["putao"],
    "青提": ["qingti", "qinti"],
    "水蜜桃": ["shuimitao"],
    "橘子": ["juzi"],
    "荔枝": ["lizhi"],
    "乳酸菌": ["rusuanjun"],
    "茉莉": ["moli"],
    "柠檬": ["ningmeng", "ningmen", "nimeng", "lemon"],
    "无糖": ["wutang"],
    "奥利奥": ["oreo"],
    "红枣": ["hongzao"],
    "椰子": ["yezi"],
    "百香果": ["baixiangguo"],
    "奇异果": ["qiyiguo"],
    "菊花": ["juhua"],
    "苹果": ["pingguo"],
    "橙": ["cheng", "chenzi"],
    "橘皮": ["jupi"],
    "乌龙": ["wulong"],
    "蜜桃": ["mitao"],
    "西柚": ["xiyou", "pomelo"],
    "番茄": ["fanqie"],
    "黄瓜": ["huanggua"],
    "烧烤": ["shaokao"],
    "牛排": ["niupai"],
    "蓝莓": ["lanmei"],
    "菠萝": ["boluo"],
    "香水柠檬": ["xiangshuiningmeng"],
    "芭乐": ["bale"],
    "莓果": ["meiguo"],
    "桃子": ["taozi"],
    "青梅": ["qingmei"],
    "陈皮": ["chenpi"],
    "香蕉": ["xiangjiao"],
    "巧克力": ["qiaokeli"],
    "淡爽": ["dan"],
    "电解质": ["dianjiezhi"],
    "杏子": ["xinzi"],
}


def parse_order_to_tasks(json_path):
    """
    解析用户订单 JSON，采用字典模糊过滤法映射到 YOLO 标签。
    核心逻辑：
    1. 根据产品名找中文大字典，圈定一个粗候选池。
    2. 如果有确定的品牌，从中剔除品牌拼音不对的。
    3. 如果有确定的口味，从中剔除口味拼音不对的。
    4. 把带有丰富属性的结构直接推给主控。
    """
    if not os.path.exists(json_path):
        print(f"⚠️ 找不到订单文件: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        order_data = json.load(f)

    requests = order_data if isinstance(order_data, list) else order_data.get("shopping_requests", [])
    active_tasks = []

    for req in requests:
        product_cn = req.get("product") or req.get("original_name", "")
        brand_cn = req.get("brand", "")
        flavor_cn = req.get("flavor", "")
        unit_cn = req.get("unit", "")
        try:
            quantity = int(req.get("count") or req.get("quantity") or 1)
        except Exception:
            quantity = 1

        if not brand_cn:
            for b_cn in BRAND_SPELLS.keys():
                if b_cn in product_cn:
                    brand_cn = b_cn
                    break

        base_candidates = set()
        combined_text = f"{brand_cn}{product_cn}{flavor_cn}"
        for key, labels in CHINESE_ITEM_DB.items():
            if key in combined_text:
                base_candidates.update(labels)

        matched_labels = list(base_candidates)

        if brand_cn:
            expected_spell_list = []
            for b_cn, b_spells in BRAND_SPELLS.items():
                if b_cn in brand_cn or b_cn in product_cn:
                    expected_spell_list.extend(b_spells)

            if expected_spell_list:
                matched_labels = [
                    label for label in matched_labels
                    if any(esp in label for esp in expected_spell_list)
                ]

        if flavor_cn:
            expected_flavor_spells = []
            for f_cn, f_spells in FLAVOR_SPELLS.items():
                if f_cn in flavor_cn or f_cn in product_cn:
                    expected_flavor_spells.extend(f_spells)

            if expected_flavor_spells:
                filtered_by_flavor = [
                    label for label in matched_labels
                    if any(fsp in label for fsp in expected_flavor_spells)
                ]
                if filtered_by_flavor:
                    matched_labels = filtered_by_flavor

        task_info = {
            "original_name": f"{brand_cn}{product_cn}{flavor_cn}",
            "product": product_cn,
            "brand": brand_cn,
            "flavor": flavor_cn,
            "unit": unit_cn,
            "matched_yolo_labels": matched_labels,
            "count": quantity,
        }

        merged = False
        for existing in active_tasks:
            if (
                existing["product"] == product_cn
                and existing["brand"] == brand_cn
                and existing["flavor"] == flavor_cn
            ):
                existing["count"] += quantity
                existing["matched_yolo_labels"] = list(
                    set(existing["matched_yolo_labels"] + matched_labels)
                )
                merged = True
                break

        if not merged:
            active_tasks.append(task_info)

    output_dir = os.path.dirname(json_path)
    debug_path = os.path.join(output_dir, "task_pool_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(active_tasks, f, ensure_ascii=False, indent=4)

    print(f"[OK] 提货映射 (detect_12 版) 解析完成！已生成清单: {debug_path}")
    return active_tasks


if __name__ == "__main__":
    test_json = os.path.join(os.path.dirname(__file__), "task_pool_input.json")
    if os.path.exists(test_json):
        print(f"Testing parse: {test_json}")
        tasks = parse_order_to_tasks(test_json)
        print(json.dumps(tasks, ensure_ascii=False, indent=4))
    else:
        print(f"未找到测试文件: {test_json}")