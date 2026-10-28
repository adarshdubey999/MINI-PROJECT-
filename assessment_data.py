"""
Assessment Data Module for Dyscalculia Screening Tool
Contains psychometrically structured mathematical assessment questions
across the 5 clinical diagnostic categories specified in Chapters 1.4, 3.3, and 7.2.3.
"""

def generate_dot_svg(count: int, pattern: str = "random", color: str = "#2563EB", size: int = 180) -> str:
    """Generate clean SVG visual dot patterns for Subitizing & Estimation tasks."""
    dots_svg = ""
    # Canonical dice patterns or clustered coordinates
    coords = {
        1: [(90, 90)],
        2: [(50, 50), (130, 130)],
        3: [(45, 45), (90, 90), (135, 135)],
        4: [(50, 50), (130, 50), (50, 130), (130, 130)],
        5: [(45, 45), (135, 45), (90, 90), (45, 135), (135, 135)],
        6: [(45, 40), (135, 40), (45, 90), (135, 90), (45, 140), (135, 140)],
        7: [(45, 40), (135, 40), (45, 90), (90, 90), (135, 90), (45, 140), (135, 140)],
        8: [(40, 40), (90, 40), (140, 40), (65, 90), (115, 90), (40, 140), (90, 140), (140, 140)],
        9: [(40, 40), (90, 40), (140, 40), (40, 90), (90, 90), (140, 90), (40, 140), (90, 140), (140, 140)],
        12: [
            (35, 35), (75, 35), (115, 35), (150, 35),
            (35, 90), (75, 90), (115, 90), (150, 90),
            (35, 145), (75, 145), (115, 145), (150, 145)
        ]
    }
    selected_coords = coords.get(count, [(40 + (i % 4) * 35, 40 + (i // 4) * 35) for i in range(count)])
    radius = 12 if count <= 6 else 9

    for cx, cy in selected_coords:
        dots_svg += f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}" />'

    return f'''
    <div style="display:flex; justify-content:center; margin:10px 0;">
        <svg width="{size}" height="{size}" viewBox="0 0 180 180" style="background:#F1F5F9; border-radius:12px; border:2px solid #CBD5E1;">
            {dots_svg}
        </svg>
    </div>
    '''


ASSESSMENT_QUESTIONS = [
    # -------------------------------------------------------------
    # Category 1: Number Recognition & Subitizing (Questions 1 to 3)
    # -------------------------------------------------------------
    {
        "id": 1,
        "category": "Number Recognition & Subitizing",
        "title": "Question 1: Rapid Dot Recognition (Subitizing)",
        "description": "Look at the visual dot pattern below. How many dots do you see in the box without counting them one-by-one?",
        "visual_type": "svg_dots",
        "dot_count": 4,
        "dot_color": "#2563EB",
        "question_text": "Select the exact number of blue dots:",
        "options": ["3", "4", "5", "6"],
        "correct_answer": "4",
        "hint": "Try to recognize the shape pattern instantly.",
        "difficulty": "Easy"
    },
    {
        "id": 2,
        "category": "Number Recognition & Subitizing",
        "title": "Question 2: Word to Number Matching",
        "description": "Read the number written in words below and pick its correct numeric symbol.",
        "visual_type": "text_card",
        "card_text": "Forty-Seven",
        "question_text": "Which digit represents 'Forty-Seven'?",
        "options": ["74", "47", "407", "40"],
        "correct_answer": "47",
        "hint": "Check the tens and units place carefully.",
        "difficulty": "Easy"
    },
    {
        "id": 3,
        "category": "Number Recognition & Subitizing",
        "title": "Question 3: Dice Pattern Recognition",
        "description": "Look at the dots arranged in the box below.",
        "visual_type": "svg_dots",
        "dot_count": 6,
        "dot_color": "#059669",
        "question_text": "How many green dots are present?",
        "options": ["5", "6", "7", "8"],
        "correct_answer": "6",
        "hint": "This is a two-column pattern of 3 pairs.",
        "difficulty": "Easy"
    },

    # -------------------------------------------------------------
    # Category 2: Numerical Comparison & Magnitude (Questions 4 to 6)
    # -------------------------------------------------------------
    {
        "id": 4,
        "category": "Numerical Magnitude Comparison",
        "title": "Question 4: Number Magnitude Comparison",
        "description": "Compare the two numbers presented below:",
        "visual_type": "comparison_boxes",
        "box_a": "83",
        "box_b": "38",
        "question_text": "Which statement is mathematically TRUE?",
        "options": ["83 is LARGER than 38", "38 is LARGER than 83", "They are equal", "Cannot be compared"],
        "correct_answer": "83 is LARGER than 38",
        "hint": "Look at the value of the digit in the tens place.",
        "difficulty": "Easy"
    },
    {
        "id": 5,
        "category": "Numerical Magnitude Comparison",
        "title": "Question 5: Distance Effect & Proximity",
        "description": "Which of the following numbers is closest in value to 100?",
        "visual_type": "target_number",
        "target": "100",
        "question_text": "Select the number closest to 100:",
        "options": ["91", "108", "96", "112"],
        "correct_answer": "96",
        "hint": "Calculate the difference between 100 and each option (100 - 96 = 4).",
        "difficulty": "Medium"
    },
    {
        "id": 6,
        "category": "Numerical Magnitude Comparison",
        "title": "Question 6: Fractional / Decimal Magnitude",
        "description": "Consider two equal-sized pizzas cut into slices.",
        "visual_type": "fraction_comparison",
        "box_a": "1/2 (Half)",
        "box_b": "1/4 (One Quarter)",
        "question_text": "Which portion is GREATER in amount?",
        "options": ["1/2 is greater than 1/4", "1/4 is greater than 1/2", "Both are identical", "1/4 is double of 1/2"],
        "correct_answer": "1/2 is greater than 1/4",
        "hint": "A half pizza is bigger than a quarter pizza.",
        "difficulty": "Medium"
    },

    # -------------------------------------------------------------
    # Category 3: Basic Arithmetic Operations (Questions 7 to 9)
    # -------------------------------------------------------------
    {
        "id": 7,
        "category": "Basic Arithmetic Operations",
        "title": "Question 7: Mental Addition",
        "description": "Solve the arithmetic equation:",
        "visual_type": "equation",
        "equation": "8 + 7 = ?",
        "question_text": "What is the sum of 8 and 7?",
        "options": ["13", "14", "15", "16"],
        "correct_answer": "15",
        "hint": "8 + 2 = 10, then add the remaining 5.",
        "difficulty": "Easy"
    },
    {
        "id": 8,
        "category": "Basic Arithmetic Operations",
        "title": "Question 8: Mental Subtraction",
        "description": "Calculate the difference:",
        "visual_type": "equation",
        "equation": "17 - 9 = ?",
        "question_text": "What is 17 minus 9?",
        "options": ["6", "7", "8", "9"],
        "correct_answer": "8",
        "hint": "17 - 10 is 7, so 17 - 9 is one more (8).",
        "difficulty": "Medium"
    },
    {
        "id": 9,
        "category": "Basic Arithmetic Operations",
        "title": "Question 9: Multiplication Fact Retrieval",
        "description": "Solve the basic multiplication fact:",
        "visual_type": "equation",
        "equation": "6 × 4 = ?",
        "question_text": "What is the product of 6 and 4?",
        "options": ["20", "22", "24", "26"],
        "correct_answer": "24",
        "hint": "Think of 6 + 6 + 6 + 6.",
        "difficulty": "Medium"
    },

    # -------------------------------------------------------------
    # Category 4: Quantity Understanding & Estimation (Questions 10 to 12)
    # -------------------------------------------------------------
    {
        "id": 10,
        "category": "Quantity Understanding & Estimation",
        "title": "Question 10: Approximate Quantity Estimation",
        "description": "Look at the array of dots below without counting one by one.",
        "visual_type": "svg_dots",
        "dot_count": 12,
        "dot_color": "#7C3AED",
        "question_text": "What is the best estimate of the number of dots?",
        "options": ["About 3 to 4", "About 10 to 14", "About 30 to 40", "More than 100"],
        "correct_answer": "About 10 to 14",
        "hint": "Look at the 3 rows of 4 dots.",
        "difficulty": "Medium"
    },
    {
        "id": 11,
        "category": "Quantity Understanding & Estimation",
        "title": "Question 11: Mental Number Line Position",
        "description": "Imagine a straight number line starting at 0 on the left and ending at 100 on the right.",
        "visual_type": "number_line",
        "start_val": 0,
        "end_val": 100,
        "target_val": 50,
        "question_text": "Where does the number 50 sit on this number line?",
        "options": ["Very close to 0", "Exactly halfway in the middle", "Very close to 100", "At the 1/4 mark"],
        "correct_answer": "Exactly halfway in the middle",
        "hint": "50 is half of 100.",
        "difficulty": "Easy"
    },
    {
        "id": 12,
        "category": "Quantity Understanding & Estimation",
        "title": "Question 12: Place Value Decomposition",
        "description": "In the number 64, what does the digit '6' represent?",
        "visual_type": "place_value",
        "number": "64",
        "question_text": "What is the value of 6 in 64?",
        "options": ["6 units (6)", "6 tens (60)", "6 hundreds (600)", "Just the digit 6"],
        "correct_answer": "6 tens (60)",
        "hint": "64 is made of 60 + 4.",
        "difficulty": "Medium"
    },

    # -------------------------------------------------------------
    # Category 5: Mathematical Reasoning & Sequences (Questions 13 to 15)
    # -------------------------------------------------------------
    {
        "id": 13,
        "category": "Mathematical Reasoning & Sequences",
        "title": "Question 13: Number Sequence Completion",
        "description": "Observe the progression pattern of numbers below:",
        "visual_type": "sequence",
        "sequence": "4, 8, 12, 16, [ ? ]",
        "question_text": "What number comes next in this sequence?",
        "options": ["18", "20", "22", "24"],
        "correct_answer": "20",
        "hint": "Notice that each number increases by 4.",
        "difficulty": "Easy"
    },
    {
        "id": 14,
        "category": "Mathematical Reasoning & Sequences",
        "title": "Question 14: Reverse Skip Counting",
        "description": "Observe the descending sequence of numbers below:",
        "visual_type": "sequence",
        "sequence": "35, 30, 25, 20, [ ? ]",
        "question_text": "What is the missing number in the sequence?",
        "options": ["10", "15", "18", "12"],
        "correct_answer": "15",
        "hint": "Each step subtracts 5.",
        "difficulty": "Easy"
    },
    {
        "id": 15,
        "category": "Mathematical Reasoning & Sequences",
        "title": "Question 15: Practical Word Problem Logic",
        "description": "Read the short scenario carefully:",
        "visual_type": "scenario",
        "text": "A teacher has 18 pencils and shares them equally among 3 students.",
        "question_text": "How many pencils does each student receive?",
        "options": ["5 pencils", "6 pencils", "7 pencils", "8 pencils"],
        "correct_answer": "6 pencils",
        "hint": "18 divided by 3.",
        "difficulty": "Medium"
    }
]


def get_all_questions() -> list:
    """Return all assessment questions."""
    return ASSESSMENT_QUESTIONS


def get_questions_by_category(category_name: str) -> list:
    """Filter questions for a specific mathematical domain."""
    return [q for q in ASSESSMENT_QUESTIONS if q['category'] == category_name]


def get_category_names() -> list:
    """Return unique ordered categories."""
    return [
        "Number Recognition & Subitizing",
        "Numerical Magnitude Comparison",
        "Basic Arithmetic Operations",
        "Quantity Understanding & Estimation",
        "Mathematical Reasoning & Sequences"
    ]
