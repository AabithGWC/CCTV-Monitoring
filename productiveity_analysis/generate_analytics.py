import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_analytics_charts(output_dir="output_vedio"):
    json_path = os.path.join(output_dir, "detection_results.json")
    csv_path = os.path.join(output_dir, "detection_timeline.csv")
    summary_path = os.path.join(output_dir, "summary_report.json")
    
    if not os.path.exists(json_path) or not os.path.exists(summary_path):
        print(f"[!] Detection results not found in {output_dir}. Please run video_detector.py first.")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df = pd.read_csv(csv_path)

    # Styling settings
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#0f172a')
    
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')

    # Subplot 1: Count Timeline
    axes[0].plot(df["Timestamp_Sec"] / 60.0, df["Person_Count"], label="Person (Count)", color="#4ade80", linewidth=2)
    axes[0].plot(df["Timestamp_Sec"] / 60.0, df["Laptop_Count"], label="Laptop (Count)", color="#38bdf8", linewidth=1.5, linestyle="--")
    axes[0].plot(df["Timestamp_Sec"] / 60.0, df["Phone_Count"], label="Mobile Phone (Count)", color="#f43f5e", linewidth=1.5)
    
    axes[0].set_title("CCTV Detection Activity Timeline (20-Minute Video)", color="white", fontsize=14, pad=12, weight="bold")
    axes[0].set_ylabel("Detected Count", color="#94a3b8", fontsize=11)
    axes[0].set_xlabel("Time (Minutes)", color="#94a3b8", fontsize=11)
    axes[0].legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='white', loc="upper right")
    axes[0].grid(True, linestyle=":", alpha=0.3, color="#64748b")

    # Subplot 2: Presence Percentages Bar Chart
    categories = ['Person Presence', 'Laptop In-Use', 'Phone Activity']
    percents = [
        summary["person_presence"]["percentage"],
        summary["laptop_presence"]["percentage"],
        summary["phone_presence"]["percentage"]
    ]
    colors = ['#4ade80', '#38bdf8', '#f43f5e']

    bars = axes[1].barh(categories, percents, color=colors, height=0.5, edgecolor='#334155')
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Percentage of Total Video Time (%)", color="#94a3b8", fontsize=11)
    axes[1].set_title("Presence & Productivity Distribution Summary", color="white", fontsize=12, pad=10)
    axes[1].grid(True, axis='x', linestyle=":", alpha=0.3, color="#64748b")

    for bar, pct in zip(bars, percents):
        axes[1].text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%",
                     va='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "analytics_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] High-resolution analytics chart saved to: {chart_path}")
    return chart_path

if __name__ == "__main__":
    generate_analytics_charts()
