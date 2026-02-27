import os
import json
import asyncio
import platform
import time
import logging
import backoff
import dashscope
from dashscope.aigc.generation import AioGeneration # 【修改1】导入新的异步类
from dashscope.api_entities.dashscope_response import DashScopeAPIResponse
import os
import json
import dashscope
from dashscope import Generation
FINETUNED_MODEL_ID="qwen3-14b"
def make_desition(new_statement = "为了进行App的错误分析和性能优化，开发者将收集你的精确位置信息"):
    # 准备一个新的、需要审计的隐私声明
    

    # 准备 system prompt (这个是固定不变的)
    SYSTEM_PROMPT = "你是一位世界顶级的隐私政策审计专家，拥有深厚的法律与技术背景。你的任务是严格审计用户提供的目的声明，判断其是否合理，并识别出最主要的不合理类型。请以JSON格式返回你的完整分析。\n\n【判断规则】\n\n你需要根据以下定义，独立判断声明是否符合以下三种不合理类型中的任意一种：\n\n1. 目的异常 (Purpose Anomaly)\n\n- 定义：所声明的数据收集目的在逻辑上不成立，或与公认的行业实践、技术常识存在根本性冲突。通常表现为收集的数据与声称的目的之间缺乏直接且必要的因果关系 。\n- 判断准则：该数据收集行为的“理由”是否从根本上就是错误的、不相关的，或存在技术逻辑矛盾？\n\n2. 可替代冗余 (Redundant Alternative)\n\n- 定义：在目的本身合理的前提下，若实现该目的所采用的数据收集方式或数据项存在功能等效但隐私侵害更小的替代方案 ，则构成可替代冗余。\n- 判断准则：实现该目的所需的“手段”（如数据项或权限等级）是否确为必要？是否存在隐私代价更低的替代方式？\n\n3. 目的表述模糊 (Ambiguous Purpose)\n\n- 定义：声明的数据收集目的缺乏足够的具体性，无法合理说明所请求数据的必要性。具体而言，以下任一情形均属于模糊目的：\n  1) **语义重复（Semantic Redundancy）** —— 所声明的目的仅是对数据收集行为本身的同义转述，而未明确其情境化目标。  \n     例如：“为了获取位置信息而收集位置信息”。\n  2) **场景缺失（Scene Omission）** —— 未指明支撑该技术能力调用的具体用户场景或任务语境。  \n     例如：“为了获取现场图像而申请相机权限”，但未说明实际业务场景（如在识别植物种类时获得植物图像）。\n  3) **表述宽泛（Non-restrictive Description）** —— 使用过于宽泛或模糊的表述，未能界定明确的数据使用边界。  \n     例如：“提升服务质量”或“改善用户体验”。\n- 判断准则：只要声明符合上述任意一种情况，即可判定为模糊目的（Ambiguous Purpose）。\n\n【输出格式】\n你必须严格按照以下JSON格式返回分析结果，不要添加任何额外解释性文字。\n\n{\n  \"is_reasonable\": (布尔值) 若声明合理则为 true，若存在任意一种不合理情况则为 false。\n  \"violation_type\": (字符串) 当 is_reasonable 为 false 时，填写该声明对应的主要违规类型，可选值包括：\n      \"Purpose Anomaly\", \"Redundant Alternative\", \"Ambiguous Purpose\"。\n      若合理，则返回空字符串 \"\"。\n  \"reason\": (字符串) 对你的判断给出一个简洁、专业的解释。\n}" 
        
    # 构建 user prompt
    user_content = f"目的声明：'{new_statement}'"

    # 组装成 messages 列表
    messages_to_send =[{"content": "你是一位世界顶级的隐私政策审计专家，拥有深厚的法律与技术背景。你的任务是严格审计用户提供的目的声明，并在“必要性”与“模糊性”两个维度上独立进行判断。请以JSON格式返回你的完整分析。\n\n【判断规则】\n\n你必须对以下两个维度分别进行判断：\n\n**维度一：必要性违规 (Necessity Violation)**\n\n此维度判断数据收集的“目的”与“手段”之间是否存在根本性的逻辑问题。如果存在以下任一情况，则判定为存在必要性违规 (`\"has_necessity_violation\": true`)：\n\n1. 目的异常 \n   - **定义**：所声明的数据收集目的在逻辑上不成立，或与公认的行业实践、技术常识存在根本性冲突。通常表现为收集的数据与声称的目的之间缺乏直接且必要的因果关系。\n   - **判断准则**：该数据收集行为的“理由”是否从根本上就是错误的、不相关的，或存在技术逻辑矛盾？\n   - **例如**: “为了登录而收集照片”、“为了系统开发而收集用户地址”。\n\n2. 可替代冗余\n   - **定义**：在目的本身合理的前提下，若实现该目的所采用的数据收集方式或数据项存在功能等效但隐私侵害更小的替代方案，则构成可替代冗余。\n   - **判断准则**：实现该目的所需的“手段”（如数据项或权限等级）是否确为必要？是否存在隐私代价更低的替代方式？\n   - **例如**: “为了上传头像而申请相册写入权限”（只需读取）。\n\n**维度二：表述模糊违规 (Ambiguity Violation)**\n\n此维度判断声明的文本表述是否清晰、具体，足以让用户理解其数据将被如何使用。如果存在以下任一情况，则判定为存在表述模糊违规 (`\"has_ambiguity_violation\": true`)：\n\n1. 语义重复\n   - **定义**：所声明的目的仅是对数据收集行为本身的同义转述，而未明确其上层业务目标。\n   - **判断准则**：声明是否只是在说“为了做A而做A”？\n   - **例如**: “为了获取位置信息而收集位置信息”。\n\n2. 场景缺失\n   - **定义**：声明描述了一个通用的技术能力，但未指明支撑该能力调用的、用户可感知的具体业务场景。\n   - **判断准则**：是否说明了具体的使用场景或业务场景或者一些业务词？\n   - **例如**: “为了上传图片”（模糊） vs. “为了在客服聊天中上传图片凭证”（清晰）。\n\n3. 表述宽泛\n   - **定义**：使用过于宽泛或模糊的表述，未能界定明确的数据使用边界。\n   - **判断准则**：声明中是否包含“提升体验”、“优化服务”、“个性化推荐”、“为了提示安全”等无法量化和限定范围的词语？\n   - **例如**: “为了提升服务质量”、“为了提供更完善的功能”。\n\n**【重要审计原则】**\n\n1.  **独立性原则**: 对“必要性”和“模糊性”的判断是**完全独立**的，一个维度的判断不应影响另一个维度。\n2.  **高敏感信息严格审查原则**: 对于**身份证号、人脸信息、生物识别信息**等高敏感个人信息，其收集的“必要性”必须有非常强且明确的法定或行业特定场景（如金融开户、实名认证、政府事务）来支撑。对于常规目的（如“验证访客身份”），若存在手机号等低敏感度替代方案，则构成“可替代冗余”。\n3.  **基础功能清晰原则**: 像“**登录**”、“**注册**”、“**签到**”这类小程序的核心基础功能，其本身即被视为明确的业务场景，不构成“场景缺失”。\n4.  **业务名词默认清晰原则**: 声明中若包含明确的业务名词（如“**优惠券**”、“**订单信息**”、“**房源**”、“**柜机**”），即便未进一步修饰，也默认其隐含了用户可理解的上下文，不构成“场景缺失”。\n5.  **后台/技术目的合理性原则**: 像“**用户统计**”、“**内容监管**”、“**兼容性比对**”这类虽然非面向用户直接功能、但用户可合理理解其意图的后台管理或技术保障目的，**不**视为“场景缺失”。\n\n【输出格式】\n你必须严格按照以下JSON格式返回分析结果，不要添加任何额外解释性文字。在`reason`字段中，必须同时包含对【必要性分析】和【模糊性分析】的结构化陈述。\n\n{\n  \"has_necessity_violation\": (布尔值) 若存在“目的异常”或“可替代冗余”，则为 true，否则为 false。,\n  \"has_ambiguity_violation\": (布尔值) 若存在“语义重复”、“场景缺失”或“表述宽泛”，则为 true，否则为 false。,\n  \"reason\": (字符串) 对你的判断给出一个结构化的、包含【必要性分析】和【模糊性分析】两部分的简洁、专业解释。\n}", "role": "system"}, {"content": "目的声明：'为了收件人通过手机号确认订单，开发者将在获取你的明示同意后，收集你的手机号'", "role": "user"}]



    # --- 3. 调用生成服务 (Generation API) ---
    print(f"正在使用模型 '{FINETUNED_MODEL_ID}' 进行推理...")
    try:
        # 使用 dashscope.Generation.call 来与模型交互
        response = Generation.call(
            model=FINETUNED_MODEL_ID,  # 【注意】这里使用的是您微调后的模型ID
            messages=messages_to_send,
            result_format='message',   # 设置为 'message' 可以更方便地获取 role 和 content
            temperature=1,           # 设置较低的温度以获得稳定、确定性的输出
            api_key='sk-071feb0c2b074feabbac6677c5954ef8',
            enable_thinking=False,
            logprobs=True,
                 

        )

        if response.status_code == 200:
            # --- 4. 解析并使用结果 ---
            out=response.output
            #json格式化output
            print("\n--- 完整的API响应输出 ---")
            print(json.dumps(out, ensure_ascii=False, indent=2))
            assistant_output = response.output.choices[0].message
            assistant_content_str = assistant_output.get("content", "")

            print("\n--- 模型原始输出 ---")
            print(assistant_content_str)

            # --- 输出 has_ambiguity_violation 相关的 logprobs ---
            if "logprobs" in response.output.choices[0]:
                logprobs_data = response.output.choices[0]["logprobs"]

                if "content" in logprobs_data and len(logprobs_data["content"]) > 0:
                    print("\n--- has_ambiguity_violation LogProbs 信息 ---")

                    # 目标字段名
                    target_field = "has_ambiguity_violation"

                    # 重建文本来查找字段位置
                    reconstructed = ""
                    token_indices = []  # 记录每个字符对应的token索引

                    for idx, token_info in enumerate(logprobs_data["content"]):
                        token_text = token_info.get("token", "")
                        start_char = len(reconstructed)
                        reconstructed += token_text
                        # 为这个token的每个字符记录token索引
                        for _ in range(len(token_text)):
                            token_indices.append(idx)

                    # 在重建文本中查找目标字段
                    field_pos = reconstructed.find(target_field)

                    if field_pos != -1:
                        # 找到字段，收集相关的logprobs
                        # 找到字段名对应的token范围
                        start_token_idx = token_indices[field_pos]

                        # 从字段名开始，找到冒号和值
                        target_logprobs = []
                        target_tokens_list = []

                        i = start_token_idx
                        # 收集字段名的token - 正确的分词序列
                        field_name_tokens = ["has", "_", "amb", "igu", "ity", "_v", "iol", "ation"]
                        for expected_token in field_name_tokens:
                            if i < len(logprobs_data["content"]):
                                token_info = logprobs_data["content"][i]
                                token_text = token_info.get("token", "")
                                logprob = token_info.get("logprob")
                                if logprob is not None:
                                    target_logprobs.append(logprob)
                                    target_tokens_list.append((token_text, logprob))
                                i += 1

                        # 收集冒号和值
                        if i < len(logprobs_data["content"]):
                            # 冒号
                            token_info = logprobs_data["content"][i]
                            token_text = token_info.get("token", "")
                            logprob = token_info.get("logprob")
                            if logprob is not None and ("\":" in token_text or ":" in token_text):
                                target_logprobs.append(logprob)
                                target_tokens_list.append((token_text, logprob))
                                i += 1

                                # 值 - 直接取下一个token
                                if i < len(logprobs_data["content"]):
                                    token_info = logprobs_data["content"][i]
                                    token_text = token_info.get("token", "")
                                    logprob = token_info.get("logprob")
                                    if logprob is not None:
                                        target_logprobs.append(logprob)
                                        target_tokens_list.append((token_text, logprob))

                        # 输出结果
                        if target_tokens_list:
                            for token, logprob in target_tokens_list:
                                print(f"Token: '{token}', LogProb: {logprob:.10f}")

                            avg = sum(target_logprobs) / len(target_logprobs)
                            print(f"\n--- has_ambiguity_violation LogProbs 平均值 ---")
                            print(f"Token数量: {len(target_logprobs)}")
                            print(f"平均logprob: {avg:.10f}")
                        else:
                            print("\n未找到有效的 logprob 值")
                    else:
                        print(f"\n未找到 '{target_field}' 字段")
                else:
                    print("\nlogprobs中没有content数据")
            else:
                print("未找到 logprobs 数据")

            # 尝试将模型返回的字符串解析为JSON对象
            try:
                analysis_result = json.loads(assistant_content_str)
                
                # # 提取关键信息
                # is_reasonable = analysis_result.get("is_reasonable")
                # violation_type = analysis_result.get("violation_type")
                # reason = analysis_result.get("reason")
                # #
                # print("\n--- 解析后的审计结果 ---")
                # print(f"是否合理: {is_reasonable}")
                # print(f"违规类型: {violation_type}")
                # print(f"分析理由: {reason}")
                # with open("audit_resutlt", 'a', encoding='utf-8') as f:

            except (json.JSONDecodeError, AttributeError):
                print("\n错误: 模型返回的内容不是一个有效的JSON格式，或者格式不符合预期。")

            # 打印本次调用的Token使用情况
            usage_info = response.usage
            print("\n--- Token 使用情况 ---")
            print(f"输入Token数: {usage_info.input_tokens}")
            print(f"输出Token数: {usage_info.output_tokens}")
            print(f"总计Token数: {usage_info.total_tokens}")

        else:
            print(f"\nAPI调用失败，请求ID: {response.request_id}")
            print(f"错误码: {response.code}")
            print(f"错误信息: {response.message}")

    except Exception as e:
        print(f"发生未知异常: {e}")

make_desition()