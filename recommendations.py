"""
Educational Guidance and Intervention Recommendations Module
Provides evidence-based strategies, classroom accommodations, and home practices
based on the student's screening risk level and primary deficit areas (Chapter 7.2.4 & 8.2).
"""

RECOMMENDATIONS_MAP = {
    "Number Recognition & Subitizing": {
        "title": "Core Subitizing & Number Sense Interventions",
        "strategies": [
            "Use Dot Plates & Ten-Frames: Practice recognizing patterns of 1 to 5 dots instantly without counting fingers.",
            "Multi-Sensory Tactile Digits: Trace sandpaper numbers or shape clay numerals to reinforce digit symbol recognition.",
            "Dice & Domino Board Games: Encourage regular board games that use standard dot dice to build automatic pattern retrieval.",
            "Disambiguate Reversible Digits: Color-code commonly confused numerals (such as 6 and 9, or 14 and 41)."
        ],
        "classroom_accommodations": [
            "Allow tactile math counters, Abacus, or Numicon tiles during calculations.",
            "Provide dot-pattern reference strips on the student's desk."
        ]
    },
    "Numerical Magnitude Comparison": {
        "title": "Numerical Magnitude & Place Value Support",
        "strategies": [
            "Cuisenaire Rods & Base-10 Blocks: Physically represent and compare unit cubes, ten-rods, and hundred-flats.",
            "Walkable Physical Number Line: Place a number line along the classroom floor so the student physically walks the distance between values.",
            "Color-Coded Place Value Columns: Use Green for Units, Blue for Tens, and Red for Hundreds to clarify place value positions.",
            "Fraction Pizza/Cake Visuals: Use circular cut-outs to demonstrate why 1/2 is physically larger than 1/4."
        ],
        "classroom_accommodations": [
            "Provide printed number line rulers (0-20 and 0-100) for all math tasks.",
            "Allow extra time for tasks requiring value comparison."
        ]
    },
    "Basic Arithmetic Operations": {
        "title": "Arithmetic Fact Retrieval & Calculation Accommodations",
        "strategies": [
            "Bridge-to-Ten Mental Strategies: Teach decomposing numbers (e.g. 8 + 7 = 8 + 2 + 5 = 10 + 5 = 15) instead of rote memorization.",
            "De-emphasize Timed Flashcard Drills: Timed drills trigger severe mathematical anxiety in dyscalculic students and inhibit working memory.",
            "Fact Family Triangles: Connect addition/subtraction and multiplication/division as related families (e.g. 6 × 4 = 24, 24 ÷ 4 = 6).",
            "Finger-Counting Transition: Respect finger counting as a legitimate coping mechanism while gradually introducing counters."
        ],
        "classroom_accommodations": [
            "Allow the use of a simple basic four-function calculator for higher-order reasoning problems.",
            "Provide graph/grid paper with large squares to help align vertical columns.",
            "Extend testing time by 25% to 50% for arithmetic examinations."
        ]
    },
    "Quantity Understanding & Estimation": {
        "title": "Spatial Quantity Estimation & Number Line Reasoning",
        "strategies": [
            "Estimation Jars: Fill jars with large beads or marbles and guide reasonable benchmark estimates (Is this closer to 10 or 100?).",
            "Empty Number Line Activities: Practice placing target numbers on an unmarked line between 0 and 100 to develop mental magnitude mapping.",
            "Everyday Measurement Tasks: Involve the student in practical cooking, measuring liquids, or checking grocery receipts.",
            "Grouping in Bundles of 10: Bundle craft sticks with rubber bands to visually feel the transition from single units to tens."
        ],
        "classroom_accommodations": [
            "Provide visual anchors on number lines (e.g. bold marks at 0, 25, 50, 75, 100).",
            "Avoid penalizing minor estimation discrepancies in non-exact problems."
        ]
    },
    "Mathematical Reasoning & Sequences": {
        "title": "Mathematical Sequencing & Word Problem Logic",
        "strategies": [
            "Highlighting Key Vocabulary: Teach students to color-code operation keywords ('total', 'sum', 'difference', 'share equally').",
            "Visual Storyboarding: Break multi-step word problems into comic-strip-like drawing boxes before writing numeric symbols.",
            "Rhythmic Skip-Counting: Practice skip counting sequences (2s, 5s, 10s) using rhythm, claps, or music.",
            "One-Step-at-a-Time Cue Cards: Provide laminated checklist cards for sequential calculation steps."
        ],
        "classroom_accommodations": [
            "Read word problems aloud or provide text-to-speech support so reading difficulties do not mask math potential.",
            "Reduce total problem count on assignments while maintaining conceptual coverage."
        ]
    }
}


GENERAL_GUIDELINES = {
    "Low Risk": {
        "badge": "Typical Numerical Processing",
        "summary": "The assessment indicates strong mathematical understanding and age-appropriate numerical processing speeds across tested domains.",
        "action_plan": [
            "Continue standard mathematics curriculum and enriched problem-solving activities.",
            "Encourage mental math challenges and logical reasoning puzzles.",
            "Monitor progress periodically during routine school evaluations."
        ]
    },
    "Moderate Risk": {
        "badge": "Mild / Borderline Mathematical Difficulties",
        "summary": "The assessment detected mild difficulties in specific mathematical sub-skills or elevated response latencies.",
        "action_plan": [
            "Implement targeted educational interventions focusing on the identified deficit areas.",
            "Use concrete manipulatives (base-10 blocks, visual number lines) for 4 to 6 weeks.",
            "Repeat the screening assessment after 6 weeks to evaluate progress.",
            "If difficulties persist despite targeted support, consider consulting an educational specialist."
        ]
    },
    "High Risk": {
        "badge": "Significant Indicators of Dyscalculia",
        "summary": "The assessment reveals persistent difficulties in core numerical processing (subitizing, magnitude comparison, or arithmetic retrieval) characteristic of developmental dyscalculia.",
        "action_plan": [
            "Formal Clinical Referral: Share this screening report with an educational psychologist, pediatrician, or special education coordinator for comprehensive clinical diagnostic evaluation (ICD-11 6A03.0 / DSM-5).",
            "Classroom Accommodations: Request formal accommodations including extra testing time (25-50%), access to calculation aids, and reduced homework volume.",
            "Specialized Math Intervention: Engage a certified learning support specialist or dyscalculia tutor utilizing multi-sensory math methodologies.",
            "Address Math Anxiety: Provide a supportive, low-pressure learning environment to rebuild mathematical confidence and self-efficacy."
        ]
    }
}


def get_recommendations_for_result(risk_outcome: str, primary_deficit: str) -> dict:
    """Generate comprehensive structured recommendations based on screening outcome."""
    general = GENERAL_GUIDELINES.get(risk_outcome, GENERAL_GUIDELINES["Moderate Risk"])

    specific = []
    for key, data in RECOMMENDATIONS_MAP.items():
        if key in primary_deficit or primary_deficit in key:
            specific.append(data)

    if not specific:
        # Default to subitizing and arithmetic if not specifically matched
        specific = [
            RECOMMENDATIONS_MAP["Number Recognition & Subitizing"],
            RECOMMENDATIONS_MAP["Basic Arithmetic Operations"]
        ]

    return {
        "general": general,
        "specific_interventions": specific,
        "disclaimer": (
            "IMPORTANT NOTICE: This screening report is generated by an Artificial Intelligence model "
            "for preliminary educational screening and awareness purposes only. It does NOT constitute a "
            "clinical diagnosis of Developmental Dyscalculia or Specific Learning Disorder. A comprehensive "
            "diagnosis requires evaluation by a licensed educational psychologist, neuropsychologist, or medical professional."
        )
    }
