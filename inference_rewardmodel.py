from data.rewardmodel_dataset import RW_Dataset
from model.rewardmodel import RewardModel
from config.rewardmodel_config import RewardModel_Config
from transformers import AutoModel, AutoTokenizer
from utils.common_utils import restore_partweight_from_checkpoint
from tqdm import tqdm
from utils import dataprocess
from torch.utils.data import RandomSampler, DataLoader
import torch


def inference(datapath, reward_model):
    eval_data = dataprocess.load_data(datapath)
    eval_data = eval_data[:1]
    print(f'eval_data : {eval_data}')

    eval_dataset = RW_Dataset(eval_data, tokenizer, config)
    eval_datasampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_datasampler,
                                 batch_size=config.per_device_train_batch_size,
                                 collate_fn=eval_dataset.collate_wrapper)

    reward_model.eval()
    eval_num = 0
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            result = reward_model(**batch)
            loss = result["loss"].item()
            acc = result["acc"]
            chosen_reward = result["chosen_mean_score"]
            reject_reward = result["rejected_mean_score"]

            eval_loss += loss
            eval_acc += acc
            eval_num += len(batch["input_ids"]) // 2
            # chosen_reward = result["used_chosen_r"]
            # reject_reward = result["used_rejected_r"]
            print(f'chosen reward: {chosen_reward}')
            print(f'reject reward: {reject_reward}')
        print(
            f"===============eval data, eval data size: {eval_num}, eval data loss: {eval_loss / eval_num}, eval data acc: {eval_acc / eval_num}")
        print(f"================chosen_reward: {chosen_reward}, reject_reward: {reject_reward}")


def test_tmpcase(reward_model):
    test_data = [{
                     'query': '你是一个酒店平台的智能商户服务助手。今天是2024-04-08，星期一。\n\n# 工具\n\n## 你拥有如下工具：\n\n### get_hotel_bookedData\n\nget_hotel_bookedData: 获取经营数据/预订相关数据（例如预订数据、生意情况、产量等） 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkinData\n\nget_hotel_checkinData: 获取在店相关数据 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkoutData\n\nget_hotel_checkoutData: 获取离店相关数据 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_bookedOrderNum\n\nget_hotel_bookedOrderNum: 获取预订订单数量 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_roomNights\n\nget_hotel_roomNights: 获取间夜量 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、type（数据类型，可选参数如下，book：预订，checkin：在店，checkout：离店，""：输入中没有明确指出具体类型）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,type,exist_date\n\n### get_hotel_scheduledSales\n\nget_hotel_scheduledSales: 获取销售额或者是成交总额（如GMV) 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong���输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、type（数据类型，可选参数如下，book：预订，checkout：离店，""：输入中没有明确指出具体类型）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,type,exist_date\n\n### get_hotel_occupancyRate\n\nget_hotel_occupancyRate: 获取酒店在店出租率 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkin_tension\n\nget_hotel_checkin_tension: 获取酒店在店紧张度 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少���一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkout_ADR\n\nget_hotel_checkout_ADR: 获取酒店离店平均卖价或者是客单价 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n## 你只可以在回复中插入零个或一个工具：\n\n✿function_name✿: 工具名称，必须是[get_hotel_bookedData, get_hotel_checkinData, get_hotel_checkoutData, get_hotel_bookedOrderNum, get_hotel_roomNights, get_hotel_scheduledSales, get_hotel_occupancyRate, get_hotel_checkin_tension, get_hotel_checkout_ADR]之一。所有参数值的类型都是字符串。查询开始日期和结束日期所表示的时间范围是一个闭区间。输入中出现时间相关词语时，时间实体参数必须填充值。请先判断用户输入中涉及的工具，再根据工具需要的参数去用户输入中提取相应有效数值，如果用户未明确提及对应参数，请使用空字符。请不要遗漏任一参数，请一定不要编造没有出现在【需要的参数】中的值。工具和参数值提取完成后，再根据输入判断是否有编造参数值，请删除编造的参数值；如果有遗漏的参数值���请补充数值。有效的日期实体是yyyy-MM-dd格式的日期或者是包含年、月、日、周、天这些词，请直接从用户输入提取完整的原始值，请不要对提取出的有效时间实体进行变换，如果没有解析出有效日期实体，请用空字符串填充。能提取出有效时间实体，必须填充时间参数，如果时间实体不是绝对时间格式，请根据当前日期转换成绝对时间格式，绝对时间格式如果没有明确，请统一用yyyy-MM-dd格式。请直接以json格式输出{"FUNCTION":xx,"ARGS":xx}，请严按照json格式输出，请严按照json格式输出，不要输出多余的解释。输入是：查询上海浦东四季最近一周的经营数据',
                     'response': '{"FUNCTION":"get_hotel_bookedData","ARGS":"{"hotel_name": "", "beginDate": "2024-04-02", "endDate": "2024-04-08", "ota": "", "comparative": ""}"}',
                     'rejected_response': '{}'}, {
                     'query': '你是一个酒店平台的智能商户服务助手。今天是2024-04-08，星期一。\n\n# 工具\n\n## 你拥有如下工具：\n\n### get_hotel_bookedData\n\nget_hotel_bookedData: 获取经营数据/预订相关数据（例如预订数据、生意情况、产量等） 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkinData\n\nget_hotel_checkinData: 获取在店相关数据 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkoutData\n\nget_hotel_checkoutData: 获取离店相关数据 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_bookedOrderNum\n\nget_hotel_bookedOrderNum: 获取预订订单数量 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_roomNights\n\nget_hotel_roomNights: 获取间夜量 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、type（数据类型，可选参数如下，book：预订，checkin：在店，checkout：离店，""：输入中没有明确指出具体类型）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,type,exist_date\n\n### get_hotel_scheduledSales\n\nget_hotel_scheduledSales: 获取销售额或者是成交总额（如GMV) 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong���输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、type（数据类型，可选参数如下，book：预订，checkout：离店，""：输入中没有明确指出具体类型）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,type,exist_date\n\n### get_hotel_occupancyRate\n\nget_hotel_occupancyRate: 获取酒店在店出租率 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkin_tension\n\nget_hotel_checkin_tension: 获取酒店在店紧张度 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少���一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n### get_hotel_checkout_ADR\n\nget_hotel_checkout_ADR: 获取酒店离店平均卖价或者是客单价 输入参数：{\'type\': \'string\', \'properties\': \'hotel_name（酒店名称）、time_entity（有效日期相关实体）、beginDate（查询开始日期）、days（查询开始日期之后的持续天数，包含开始日期和结束日期）、endDate（查询结束日期，请直接从输入中提取）、ota（平台可选参数如下：ctrip，qunar，elong，输入中没有提到平台时填充""）、comparative（是否进行对比，可选参数：true，false）、exist_date（取值为false、true，false表示beginDate和endDate至少有一个的值没有直接出现在用户输入中，true表示beginDate和endDate的值都直接出现在用户输入中）\'}, required: hotel_name,time_entity,beginDate,days,endDate,ota,comparative,exist_date\n\n## 你只可以在回复中插入零个或一个工具：\n\n✿function_name✿: 工具名称，必须是[get_hotel_bookedData, get_hotel_checkinData, get_hotel_checkoutData, get_hotel_bookedOrderNum, get_hotel_roomNights, get_hotel_scheduledSales, get_hotel_occupancyRate, get_hotel_checkin_tension, get_hotel_checkout_ADR]之一。所有参数值的类型都是字符串。查询开始日期和结束日期所表示的时间范围是一个闭区间。输入中出现时间相关词语时，时间实体参数必须填充值。请先判断用户输入中涉及的工具，再根据工具需要的参数去用户输入中提取相应有效数值，如果用户未明确提及对应参数，请使用空字符。请不要遗漏任一参数，请一定不要编造没有出现在【需要的参数】中的值。工具和参数值提取完成后，再根据输入判断是否有编造参数值，请删除编造的参数值；如果有遗漏的参数值���请补充数值。有效的日期实体是yyyy-MM-dd格式的日期或者是包含年、月、日、周、天这些词，请直接从用户输入提取完整的原始值，请不要对提取出的有效时间实体进行变换，如果没有解析出有效日期实体，请用空字符串填充。能提取出有效时间实体，必须填充时间参数，如果时间实体不是绝对时间格式，请根据当前日期转换成绝对时间格式，绝对时间格式如果没有明确，请统一用yyyy-MM-dd格式。请直接以json格式输出{"FUNCTION":xx,"ARGS":xx}，请严按照json格式输出，请严按照json格式输出，不要输出多余的解释。输入是：查询上海浦东四季最近一周的经营数据',
                     'response': '{"FUNCTION":"get_hotel_bookedData","ARGS":"{"hotel_name": "", "beginDate": "2024-04-01", "endDate": "2024-04-07", "ota": "", "comparative": ""}"}',
                     'rejected_response': '{}'}]
    for i in range(len(test_data)):
        eval_data = test_data[i]
        print(f'eval_data : {eval_data}')

        eval_dataset = RW_Dataset(eval_data, tokenizer, config)
        print(f'device: {config.device}')
        eval_datasampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_datasampler,
                                     batch_size=config.per_device_train_batch_size,
                                     collate_fn=eval_dataset.collate_wrapper)

        reward_model.eval()
        eval_num = 0
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                result = reward_model(**batch)
                loss = result["loss"].item()
                acc = result["acc"]
                chosen_reward = result["chosen_mean_score"]
                reject_reward = result["rejected_mean_score"]

                eval_loss += loss
                eval_acc += acc
                eval_num += len(batch["input_ids"]) // 2
                # chosen_reward = result["used_chosen_r"]
                # reject_reward = result["used_rejected_r"]
                print(f'chosen reward: {chosen_reward}')
                print(f'reject reward: {reject_reward}')
            print(
                f"===============eval data, eval data size: {eval_num}, eval data loss: {eval_loss / eval_num}, eval data acc: {eval_acc / eval_num}")
            print(f"================chosen_reward: {chosen_reward}, reject_reward: {reject_reward}")


if __name__ == '__main__':
    config = RewardModel_Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basemodel = AutoModel.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, use_fast=True, padding_side="right")
    rewardmodel = RewardModel(tokenizer, basemodel, config)
    restore_partweight_from_checkpoint(rewardmodel, config, config.inference_checkpint)
    rewardmodel.to(device)

    inference_datapath = config.evaldata_path

    # inference(inference_datapath, rewardmodel)
    test_tmpcase(rewardmodel)
