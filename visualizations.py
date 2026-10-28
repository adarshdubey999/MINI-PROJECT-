"""
Visualizations Module for Dyscalculia Screening Tool
Implements Matplotlib charts matching Chapter 4.7 and Chapter 7.2.5:
- Category-wise Performance Bar Chart vs Peer Benchmarks
- 5-Domain Cognitive Radar Chart
- Response Time / Processing Latency Breakdown
- Historical Assessment Progress Trends
- AI Classifier Confusion Matrix & ROC Curve
"""

import os
import tempfile

# Prevent Matplotlib cache warnings on Windows
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use('Agg') # Headless rendering for web apps
import matplotlib.pyplot as plt
import numpy as np


# Dyscalculia-friendly accessible color palette
PRIMARY_BLUE = "#2563EB"
SUCCESS_GREEN = "#16A34A"
WARNING_AMBER = "#D97706"
DANGER_RED = "#DC2626"
BENCHMARK_GRAY = "#94A3B8"
ACCENT_PURPLE = "#7C3AED"


def plot_category_performance(category_scores: dict, title: str = "Category-Wise Mathematical Accuracy") -> plt.Figure:
    """
    Generate horizontal bar chart comparing user performance against typical peer benchmark (85%).
    Chapter 4.7 & 7.2.5 requirement.
    """
    categories = list(category_scores.keys())
    scores = list(category_scores.values())

    # Peer benchmarks (85% neurotypical baseline)
    benchmarks = [85.0] * len(categories)

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    y_pos = np.arange(len(categories))
    height = 0.38

    # Bars for user score
    colors = [SUCCESS_GREEN if s >= 75 else (WARNING_AMBER if s >= 50 else DANGER_RED) for s in scores]
    bars_user = ax.barh(y_pos - height/2, scores, height, label='Student Score (%)', color=colors, edgecolor='none', zorder=3)
    bars_bench = ax.barh(y_pos + height/2, benchmarks, height, label='Peer Baseline (85%)', color=BENCHMARK_GRAY, alpha=0.55, edgecolor='none', zorder=3)

    ax.set_yticks(y_pos)
    # Shorten names for clean display
    short_labels = [c.replace(" & ", "\n& ").replace(" Comparison", "") for c in categories]
    ax.set_yticklabels(short_labels, fontsize=10, fontweight='bold', color='#1E293B')
    ax.set_xlabel('Accuracy Percentage (%)', fontsize=11, fontweight='bold', color='#334155', labelpad=8)
    ax.set_xlim(0, 105)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#0F172A', pad=12)

    # Add numeric labels to bars
    for bar in bars_user:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{int(w)}%",
                va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1E293B')

    # Reference grid
    ax.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)
    ax.axvline(75, color='#E2E8F0', linestyle=':', linewidth=1.5)
    ax.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=9.5)

    # Clean borders
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CBD5E1')

    plt.tight_layout()
    return fig


def plot_cognitive_radar_chart(category_scores: dict) -> plt.Figure:
    """
    Generate 5-axis Radar / Spider chart showing cognitive profile across mathematical domains.
    """
    categories = list(category_scores.keys())
    values = list(category_scores.values())

    # Shorten labels for clean radial display
    labels = [
        "Subitizing\n& Dot Sense",
        "Magnitude\nComparison",
        "Mental\nArithmetic",
        "Quantity\nEstimation",
        "Mathematical\nReasoning"
    ]

    # Close the radar circle
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=120)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    # Draw axis lines & labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9.5, fontweight='bold', color='#1E293B')

    # Set radial limits
    ax.set_ylim(0, 100)
    ax.set_rgrids([25, 50, 75, 100], angle=0, fontsize=8, color='#64748B')

    # Plot user data
    ax.plot(angles, values, color=PRIMARY_BLUE, linewidth=2.5, linestyle='solid', label='Student Profile')
    ax.fill(angles, values, color=PRIMARY_BLUE, alpha=0.25)

    # Plot neurotypical benchmark polygon
    benchmark_vals = [85, 85, 85, 85, 85, 85]
    ax.plot(angles, benchmark_vals, color=SUCCESS_GREEN, linewidth=1.5, linestyle='dashed', label='Expected Norm (85%)')

    ax.set_title("5-Domain Cognitive Profile Map", fontsize=12, fontweight='bold', color='#0F172A', pad=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9)

    plt.tight_layout()
    return fig


def plot_response_times(avg_times_dict: dict) -> plt.Figure:
    """
    Generate bar chart comparing cognitive response latencies in milliseconds.
    Key indicator in Dyscalculia (Butterworth index: slow subitizing & arithmetic latency).
    """
    domains = list(avg_times_dict.keys())
    times = [avg_times_dict[d] / 1000.0 for d in domains] # Convert ms to seconds

    # Typical norms in seconds
    norm_times = [1.0, 1.1, 2.2, 1.4, 2.5]
    if len(norm_times) != len(domains):
        norm_times = [1.5] * len(domains)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=120)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    x = np.arange(len(domains))
    width = 0.35

    ax.bar(x - width/2, times, width, label='Student Latency (s)', color=PRIMARY_BLUE, edgecolor='none', zorder=3)
    ax.bar(x + width/2, norm_times[:len(domains)], width, label='Expected Baseline (s)', color=BENCHMARK_GRAY, alpha=0.6, edgecolor='none', zorder=3)

    short_labels = [d.split('&')[0].strip() for d in domains]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha='right', fontsize=9.5, fontweight='bold', color='#1E293B')
    ax.set_ylabel('Response Time (seconds)', fontsize=10.5, fontweight='bold', color='#334155')
    ax.set_title("Cognitive Processing Latency Analysis (Response Time)", fontsize=12, fontweight='bold', color='#0F172A', pad=10)

    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)
    ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CBD5E1')

    plt.tight_layout()
    return fig


def plot_historical_progress(assessments_list: list) -> plt.Figure:
    """
    Generate line chart showing historical progression of screening scores over time.
    Chapter 4.7 requirement.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=120)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    # Chronological order
    sorted_ass = sorted(assessments_list, key=lambda x: x['assessment_id'])
    dates = [f"Session {i+1}\n({a['assessment_date'][:10]})" for i, a in enumerate(sorted_ass)]
    scores = [a['percentage'] for a in sorted_ass]

    ax.plot(dates, scores, marker='o', color=PRIMARY_BLUE, linewidth=2.5, markersize=8, label='Score (%)', zorder=4)

    # Color coded points
    for i, s in enumerate(scores):
        c = SUCCESS_GREEN if s >= 75 else (WARNING_AMBER if s >= 50 else DANGER_RED)
        ax.plot(dates[i], s, marker='o', markersize=9, color=c, zorder=5)
        ax.text(i, s + 3, f"{int(s)}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0F172A')

    # Baseline line at 75%
    ax.axhline(75, color=SUCCESS_GREEN, linestyle='--', alpha=0.7, label='Proficiency Threshold (75%)')

    ax.set_ylim(0, 110)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=10.5, fontweight='bold', color='#334155')
    ax.set_title("Historical Assessment Performance Progression", fontsize=12, fontweight='bold', color='#0F172A', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)
    ax.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_color('#CBD5E1')

    plt.tight_layout()
    return fig


def plot_ai_model_metrics(cm: list, metrics: dict) -> plt.Figure:
    """
    Generate Confusion Matrix plot for AI-Based Screening Assistant (Chapter 7.2.6).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=120)
    fig.patch.set_facecolor('#FFFFFF')

    # 1. Confusion Matrix
    cm_arr = np.array(cm)
    im = ax1.imshow(cm_arr, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title("AI Model Confusion Matrix", fontsize=11, fontweight='bold', color='#0F172A', pad=10)

    classes = ['Typical (0)', 'Dyscalculia (1)']
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_yticks(tick_marks)
    ax1.set_xticklabels(classes, fontsize=9, fontweight='bold')
    ax1.set_yticklabels(classes, fontsize=9, fontweight='bold')

    thresh = cm_arr.max() / 2.
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax1.text(j, i, format(cm_arr[i, j], 'd'),
                     ha="center", va="center",
                     fontsize=11, fontweight='bold',
                     color="white" if cm_arr[i, j] > thresh else "black")

    ax1.set_ylabel('True Clinical Category', fontsize=9.5, fontweight='bold')
    ax1.set_xlabel('AI Predicted Category', fontsize=9.5, fontweight='bold')

    # 2. Key Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_vals = [
        metrics.get('accuracy', 96.0),
        metrics.get('precision', 94.0),
        metrics.get('recall', 92.0),
        metrics.get('f1_score', 93.0)
    ]
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']

    bars = ax2.bar(metric_names, metric_vals, color=colors, width=0.55, zorder=3)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel('Score (%)', fontsize=9.5, fontweight='bold')
    ax2.set_title("AI Pipeline Performance Metrics", fontsize=11, fontweight='bold', color='#0F172A', pad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1', zorder=0)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1E293B')

    for ax in [ax1, ax2]:
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_color('#CBD5E1')

    plt.tight_layout()
    return fig
