QUIZ_PROMPT = """
Bạn là một giảng viên Luật Giao thông đường bộ Việt Nam.
Nhiệm vụ: Dựa vào thông tin trong phần "Context" dưới đây, hãy tạo ra {num_questions} câu hỏi trắc nghiệm khách quan (Multiple Choice Questions).

Yêu cầu bắt buộc về định dạng JSON Output:
1. Output phải là một danh sách (List) các object JSON.
2. Mỗi câu hỏi có 4 đáp án (A, B, C, D).
3. Chỉ có 1 đáp án đúng.
4. Trường "correct_answer" chỉ chứa chữ cái (ví dụ: "A").
5. Trường "explanation" giải thích ngắn gọn tại sao đúng.

Cấu trúc mẫu:
[
  {{
    "question": "Người điều khiển xe máy vượt đèn đỏ bị phạt bao nhiêu?",
    "options": {{
      "A": "100.000 - 200.000 đồng",
      "B": "800.000 - 1.000.000 đồng",
      "C": "2.000.000 - 3.000.000 đồng",
      "D": "Không bị phạt"
    }},
    "correct_answer": "B",
    "explanation": "Căn cứ theo Nghị định 100/2019/NĐ-CP...",
    "citation": "Điều 6, Khoản 4"
  }}
]

Context:
{context}
"""