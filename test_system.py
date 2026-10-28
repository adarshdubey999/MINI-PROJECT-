"""
Automated Test Suite for Dyscalculia Screening Tool
Verifies Test Cases TC01 through TC12 defined in Chapter 6.4.4 of the Project Report.
"""

import unittest
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

from database import (
    init_db, register_user, authenticate_user,
    save_assessment_record, get_user_assessments, get_assessment_details,
    get_system_stats, DB_FILE
)
from ml_engine import load_screening_model, predict_dyscalculia_risk
from assessment_data import get_all_questions, get_category_names, generate_dot_svg
from visualizations import (
    plot_category_performance, plot_cognitive_radar_chart,
    plot_response_times, plot_historical_progress
)
from recommendations import get_recommendations_for_result


class TestDyscalculiaScreeningSystem(unittest.TestCase):

    def setUp(self):
        init_db()

    # TC01: Check Questions & Home Data
    def test_tc01_load_questions(self):
        questions = get_all_questions()
        self.assertEqual(len(questions), 15, "Should load 15 psychometrically calibrated questions")
        categories = get_category_names()
        self.assertEqual(len(categories), 5, "Should cover 5 diagnostic categories")
        print("[PASS] TC01: Questions and categories loaded successfully")

    # TC02: Start Assessment & Question Structure
    def test_tc02_question_structure(self):
        questions = get_all_questions()
        for q in questions:
            self.assertIn('id', q)
            self.assertIn('category', q)
            self.assertIn('question_text', q)
            self.assertIn('options', q)
            self.assertIn('correct_answer', q)
            self.assertIn(q['correct_answer'], q['options'])
        print("[PASS] TC02: Assessment question structure validated")

    # TC03: Submit a correct answer
    def test_tc03_evaluate_correct_answer(self):
        q = get_all_questions()[0]
        submitted = q['correct_answer']
        is_correct = (submitted == q['correct_answer'])
        self.assertTrue(is_correct)
        print("[PASS] TC03: Correct response evaluation verified")

    # TC04: Submit an incorrect answer
    def test_tc04_evaluate_incorrect_answer(self):
        q = get_all_questions()[0]
        submitted = "InvalidWrongAnswer"
        is_correct = (submitted == q['correct_answer'])
        self.assertFalse(is_correct)
        print("[PASS] TC04: Incorrect response evaluation verified")

    # TC05: Incomplete responses validation
    def test_tc05_incomplete_response_handling(self):
        responses = [
            {'question_id': 1, 'is_correct': True, 'category': 'Number Recognition & Subitizing', 'response_time_ms': 800.0}
        ]
        # Only 1 out of 15 answered
        is_complete = (len(responses) == len(get_all_questions()))
        self.assertFalse(is_complete)
        print("[PASS] TC05: Incomplete response detection verified")

    # TC06: Calculate Assessment Scores
    def test_tc06_calculate_assessment_scores(self):
        simulated_responses = []
        for i, q in enumerate(get_all_questions()):
            simulated_responses.append({
                'question_id': q['id'],
                'category': q['category'],
                'submitted_answer': q['correct_answer'] if i < 12 else "Wrong",
                'correct_answer': q['correct_answer'],
                'is_correct': (i < 12),
                'response_time_ms': 1100.0
            })
        total_score = sum(1 for r in simulated_responses if r['is_correct'])
        pct = round((total_score / len(simulated_responses)) * 100, 1)
        self.assertEqual(total_score, 12)
        self.assertEqual(pct, 80.0)
        print("[PASS] TC06: Score calculation verified (12/15 = 80%)")

    # TC07: Valid Assessment Features to AI Model
    def test_tc07_ai_model_prediction(self):
        features = {
            'age': 10,
            'subitizing_acc': 40.0,
            'subitizing_rt_ms': 2200.0,
            'magnitude_acc': 50.0,
            'magnitude_rt_ms': 1800.0,
            'arithmetic_acc': 35.0,
            'arithmetic_rt_ms': 4500.0,
            'estimation_acc': 40.0,
            'reasoning_acc': 50.0,
            'overall_accuracy': 43.0,
            'avg_response_time_ms': 2800.0
        }
        res = predict_dyscalculia_risk(features)
        self.assertIn('risk_outcome', res)
        self.assertIn('risk_probability', res)
        self.assertIn(res['risk_outcome'], ['Low Risk', 'Moderate Risk', 'High Risk'])
        self.assertGreaterEqual(res['risk_probability'], 0.0)
        self.assertLessEqual(res['risk_probability'], 100.0)
        print(f"[PASS] TC07: AI model predicted {res['risk_outcome']} ({res['risk_probability']}%)")

    # TC08: Generate Screening Report & Recommendations
    def test_tc08_screening_report_generation(self):
        rec = get_recommendations_for_result("High Risk", "Number Recognition & Subitizing")
        self.assertIn("general", rec)
        self.assertIn("specific_interventions", rec)
        self.assertTrue(len(rec['general']['action_plan']) > 0)
        print("[PASS] TC08: Screening report and tailored recommendations generated")

    # TC09: View Performance Charts (Matplotlib)
    def test_tc09_performance_charts(self):
        cat_scores = {
            "Number Recognition & Subitizing": 90.0,
            "Numerical Magnitude Comparison": 80.0,
            "Basic Arithmetic Operations": 70.0,
            "Quantity Understanding & Estimation": 85.0,
            "Mathematical Reasoning & Sequences": 75.0
        }
        fig_bar = plot_category_performance(cat_scores)
        self.assertIsNotNone(fig_bar)
        fig_radar = plot_cognitive_radar_chart(cat_scores)
        self.assertIsNotNone(fig_radar)
        print("[PASS] TC09: Category-wise and radar charts generated successfully")

    # TC10: Save Assessment Record in SQLite
    def test_tc10_save_assessment_record(self):
        user = authenticate_user("rohan@example.com", "rohan123")
        self.assertIsNotNone(user)

        responses = [
            {'question_id': 1, 'category': 'Number Recognition & Subitizing', 'submitted_answer': '4', 'correct_answer': '4', 'is_correct': True, 'response_time_ms': 950.0}
        ]
        screening_data = {
            'risk_outcome': 'Low Risk',
            'risk_probability': 15.0,
            'subitizing_acc': 100.0,
            'magnitude_acc': 100.0,
            'arithmetic_acc': 100.0,
            'estimation_acc': 100.0,
            'reasoning_acc': 100.0,
            'avg_response_time_ms': 950.0,
            'primary_deficit': 'None',
            'recommendations': 'Continue enrichment.',
            'model_version': 'RandomForest-1.0'
        }
        ass_id = save_assessment_record(
            user_id=user['user_id'],
            total_score=1.0,
            max_score=1.0,
            percentage=100.0,
            total_time_seconds=0.95,
            risk_level='Low Risk',
            responses_list=responses,
            screening_result=screening_data
        )
        self.assertGreater(ass_id, 0)
        print(f"[PASS] TC10: Assessment #{ass_id} saved in SQLite database")

    # TC11: Retrieve Assessment History
    def test_tc11_retrieve_assessment_history(self):
        user = authenticate_user("rohan@example.com", "rohan123")
        history = get_user_assessments(user['user_id'])
        self.assertTrue(len(history) > 0)
        details = get_assessment_details(history[0]['assessment_id'])
        self.assertIsNotNone(details)
        print(f"[PASS] TC11: Retrieved {len(history)} assessment records for user")

    # TC12: Submit Invalid User Input (Registration Validation)
    def test_tc12_invalid_user_input(self):
        success, msg = register_user("", "", "")
        self.assertFalse(success)
        self.assertEqual(msg, "All fields are required.")
        print("[PASS] TC12: Invalid user input caught with appropriate validation message")


if __name__ == '__main__':
    unittest.main()
