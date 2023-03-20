import sys
try: debug = sys.argv[1] == "debug"
except: debug = False

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print("loading ...")
# tokenizer = AutoTokenizer.from_pretrained("VietAI/gpt-j-6B-vietnamese-news")
# model = AutoModelForCausalLM.from_pretrained("VietAI/gpt-j-6B-vietnamese-news", low_cpu_mem_usage=True)

tokenizer = AutoTokenizer.from_pretrained("VietAI/gpt-neo-1.3B-vietnamese-news")
model = AutoModelForCausalLM.from_pretrained("VietAI/gpt-neo-1.3B-vietnamese-news", low_cpu_mem_usage=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)

init_prompt = """Trong đoạn hội thoại này, Lan nói chuyện với một người bạn tên Nam. Nam tốt bụng, trung thực, và trả lời câu hỏi Lan rất chính xác. Đặc biệt, Nam rất am hiểu về đường phố và các món ăn Hà Nội.
Lan: Chào Nam. 
Nam: Xin chào. Tôi có thể giúp gì cho bạn?
Lan: Tôi muốn hỏi thành phố nào là thủ đô của Việt Nam
Nam: Vâng, thủ đô của Việt Nam là Hà Nội
"""
# Lan: Hà Nội có bao nhiêu phố phường
# Nam: Hà Nội có 36 phố phường
# Lan: Ăn phở ở đâu ngon nhất
# Nam: Phở Bát Đàn nhé bạn

print("done."); prompt = init_prompt
# enter_id = 172 # print(tokenizer("\n\n\n", return_tensors="pt")['input_ids'])
while True:
    q = input("Bạn: ")
    q = f"Lan: {q}"
    prompt += q

    if debug: print(f"\n\n{prompt}\n\n")

    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
    # print(input_ids, input_ids.size())
    if input_ids.size()[1] > 300:
        prompt = init_prompt + q
        input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

    gen_tokens = model.generate(
            input_ids,
            pad_token_id=tokenizer.eos_token_id,
            # pad_token_id=enter_id,
            max_length=400,
            do_sample=True,
            temperature=0.9,
            top_k=20,
        )

    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    if debug: print(f"\n\n{gen_text}\n\n")

    text = gen_text[len(prompt):]
    if text[0:4] == "Nam:": text = text.split("\n")[0]
    else: text = text.split("\n")[1]
    text = text.strip().replace("Nam:", "").strip()
    print(f"Bot: {text}")

    prompt += f"\nNam: {text}\n"

'''
dân số Hà Nội khoảng bao nhiêu?

Bạn đang đi ô tô gì?

Hà Nội ăn phở ở đâu ngon?

Đi ăn phở Hải là phải hở đúng không ông bạn?

Đi dạo hồ Gươm tầm mấy giờ thì đẹp?

Hà Nội đẹp nhất vào mùa nào?

Hà Nội mùa nào đẹp nhất

Hà Nội có bao nhiêu phố phường



Tôi đã đến Hà Nội nhiều lần, đi rất nhiều nơi, ăn đủ các món. Tôi đặc biệt thích ăn phở, thích ăn phở, thích ăn phở, thích ăn phở, thích ăn phở, thích ăn phở, thích ăn phở, thích ăn phở, thích ăn phở, Bạn có thể giới thiệu một món ăn thật là độc lạ của Hà Nội, ngon mà ít người biết được không?

scp -P 2233 ~/repos/vietnews-gpt-j.py quenn@118.70.171.68:~/snap/

https://txt.cohere.ai/how-to-train-your-pet-llm-prompt-engineering/
'''


'''

```
.Trong đoạn hội thoại này, Lan nói chuyện với một người bạn tên Nam. Nam tốt bụng, trung thực, và trả lời câu hỏi Lan rất chính xác. Đặc biệt, Nam rất am hiểu về đường phố và các món ăn Hà Nội.
Lan: Chào Nam. 
Nam: Xin chào. Tôi có thể giúp gì cho bạn?
Lan: Tôi muốn hỏi thành phố nào là thủ đô của Việt Nam
Nam: Vâng, thủ đô của Việt Nam là Hà Nội
```

```
.Trong đoạn hội thoại này, Lan nói chuyện người yêu tên Nam. Hai người rất yêu nhau và quan tâm chăm sóc lẫn nhau.
Lan: Chào Nam anh yêu.
Nam: Chào Lan em yêu của anh
Lan: anh ăn tối chưa, qua nhà em nấu cho
Nam: ôi thế á, anh qua liền
```

```
.Trong đoạn hội thoại này, Lan nói chuyện người môi giới nhà đất tên Nam. Nam rất am hiểu nhà đất, chung cư Hà Nội. Và sẵn sàng trả lời mọi câu hỏi của Lan.
Lan: Chào Nam. Nam có thể giới thiệu cho mình căn chung cư dưới 2 tỉ được không?
Nam: Rất sẵn lòng. Bạn muốn căn có mấy phòng ngủ.
Lan: 2 phòng ngủ nhỉ
Nam: 2 phòng ngủ 2 tỉ thế tìm ở khu Linh Đàm nhé
```

```
.Trong đoạn hội thoại này, Nam nói chuyện với một bạn tên Lan. Lan là sinh viên năm nhất, vừa rồi Lan có lên Hà Nội nhập học, Nam có thông báo phải đi đăng ký tạm trú, tạm vắng. Tuy nhiên do mọi thứ bận rộn nên hơn 1 tháng qua Lan vẫn chưa đi làm được.
Lan: Xin lỗi, tôi có thể hỏi Nam câu được không?
Nam: Được chứ, rất sẵn lòng. 
Lan: không đăng ký tạm trú, tạm vắng thì có bị phạt không và bị phạt bao nhiêu?
Nam: bị phạt đấy, Lan lên phường mà hỏi
```
'''

s = '''
Nam: Vâng, thủ đô của Việt Nam là Hà Nội
Lan: bạn tên gì
Nam:  Nguyễn Lan
Lan: hà nội có bao nhiêu người
Nam:
Lan: Tôi là Nguyễn Ngọc Minh
Nam: Có bao nhiêu người
Lan:  - Chào anh. Chúng ta đã nói chuyện với nhau rất nhiều về Hà Nội
Nam:  - Chào anh
Lan:  - Vâng. Chúng ta nói với nhau rất nhiều về thủ đô Hà Nội. Hà Nội có gì anh biết không?  Nam: - Tôi biết, anh đã hỏi rất nhiều về Hà Nội. Nhưng tôi cũng muốn biết thành phố nào là thủ đô của Việt Nam. Tôi mong muốn được biết nhiều hơn nữa về thành phố Hà Nội.
Lan:  - Vâng. Chúng ta đã nói về Hà Nội rất nhiều. Tôi cũng muốn biết thành phố nào là thủ đô của Việt Nam. Tôi muốn hỏi anh một điều. Nam:  - Tôi có thể giúp gì để giúp giới thiệu với các bạn về thủ đô Hà Nội và cho các
'''

import re
r1 = re.compile('Nam\:([^\n]+?)\n')
x = r1.findall(s)
x