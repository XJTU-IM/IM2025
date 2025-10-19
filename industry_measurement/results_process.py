def merge_results(results1, results2=None): #results2=None 表示测试
    # 合并两个结果列表，并按指定格式返回字符串

    if results2 is None:
        output_text = "START\n"
        for result in results1:
            line = f"Goal_ID={result['Goal_ID']};Goal_A={result['Goal_A']:.1f};Goal_B={result['Goal_B']:.1f};Goal_C={result['Goal_C']:.1f};Goal_D={result['Goal_D']:.1f}\n"
            output_text += line
        output_text += "END"
        return output_text

    else:
        all_results = results1 + results2

        output_text = "START\n"
        for result in all_results:
            line = f"Goal_ID={result['Goal_ID']};Goal_A={result['Goal_A']:.1f};Goal_B={result['Goal_B']:.1f};Goal_C={result['Goal_C']:.1f};Goal_D={result['Goal_D']:.1f}\n"
            output_text += line
        output_text += "END"

        return output_text