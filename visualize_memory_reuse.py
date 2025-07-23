#!/usr/bin/env python3
"""
Visualization script for memory reuse efficiency in GenoMAS.
This script creates professional, information-dense plots showing:
1. Cumulative time savings through code reuse
2. Reuse patterns by action unit
3. Memory efficiency over problem progression
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Custom color schemes
COLOR_SCHEME = {
    'direct_reuse': '#2ecc71',      # Green - direct reuse
    'adapted_reuse': '#3498db',     # Blue - adapted reuse  
    'new_generation': '#e74c3c',    # Red - new generation
    'time_saved': '#27ae60',        # Dark green - time saved
    'time_total': '#95a5a6',        # Gray - total time
}

AGENT_COLORS = {
    'geo_agent': '#e74c3c',         # Red
    'tcga_agent': '#3498db',        # Blue
    'statistician_agent': '#f39c12', # Orange
}


def load_memory_data(data_dir: str = "./output/memory_tracking") -> List[Dict]:
    """Load all memory tracking data from the output directory"""
    json_files = glob.glob(os.path.join(data_dir, "memory_tracking_*.json"))
    all_data = []
    
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_data.append(data)
    
    return all_data


def create_cumulative_efficiency_plot(data: List[Dict], output_path: str):
    """Create stacked area chart showing cumulative time savings"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
    
    # Process data for all sessions
    for session_idx, session_data in enumerate(data):
        timeline = session_data['statistics']['reuse_efficiency_timeline']
        if not timeline:
            continue
            
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(timeline)
        
        # Group by problem index to aggregate data
        grouped = df.groupby('problem_index').agg({
            'cumulative_time_saved': 'last',
            'cumulative_time_total': 'last',
            'efficiency_ratio': 'last'
        }).reset_index()
        
        # Plot cumulative times as stacked areas
        ax1.fill_between(grouped['problem_index'], 0, grouped['cumulative_time_saved'], 
                        alpha=0.7, color=COLOR_SCHEME['time_saved'], 
                        label='Time Saved Through Reuse' if session_idx == 0 else "")
        ax1.fill_between(grouped['problem_index'], grouped['cumulative_time_saved'], 
                        grouped['cumulative_time_total'], alpha=0.5, 
                        color=COLOR_SCHEME['time_total'],
                        label='Additional Time for New Generation' if session_idx == 0 else "")
        
        # Plot efficiency ratio
        ax2.plot(grouped['problem_index'], grouped['efficiency_ratio'] * 100, 
                linewidth=2, marker='o', markersize=4, alpha=0.8,
                label=f'Session {session_idx + 1}')
        
        # Add annotations for key milestones
        milestones = [10, 25, 50, 75, 100]
        for milestone in milestones:
            if milestone < len(grouped):
                efficiency = grouped.iloc[milestone]['efficiency_ratio'] * 100
                if efficiency > 10:  # Only annotate significant savings
                    ax1.annotate(f'{efficiency:.0f}%', 
                               xy=(milestone, grouped.iloc[milestone]['cumulative_time_saved']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.8)
    
    # Customize main plot
    ax1.set_xlabel('Problem Index', fontsize=12)
    ax1.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax1.set_title('Memory Reuse Efficiency: Cumulative Time Savings', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Customize efficiency ratio plot
    ax2.set_xlabel('Problem Index', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Reuse Efficiency Ratio Over Time', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_action_unit_reuse_patterns(data: List[Dict], output_path: str):
    """Create heatmap showing reuse patterns by action unit"""
    # Aggregate data across all sessions
    action_unit_stats = {}
    
    for session_data in data:
        patterns = session_data['statistics']['reuse_patterns']
        for key, stats in patterns.items():
            if key not in action_unit_stats:
                action_unit_stats[key] = {
                    'direct_reuse_rates': [],
                    'adapted_reuse_rates': [],
                    'new_generation_rates': [],
                    'avg_time_saved': [],
                    'total_executions': 0
                }
            action_unit_stats[key]['direct_reuse_rates'].append(stats['direct_reuse_rate'])
            action_unit_stats[key]['adapted_reuse_rates'].append(stats['adapted_reuse_rate'])
            action_unit_stats[key]['new_generation_rates'].append(stats['new_generation_rate'])
            action_unit_stats[key]['avg_time_saved'].append(stats['avg_time_saved_per_execution'])
            action_unit_stats[key]['total_executions'] += stats['total_executions']
    
    # Calculate averages
    action_units = []
    reuse_data = {'Direct Reuse': [], 'Adapted Reuse': [], 'New Generation': []}
    time_saved_data = []
    
    for key, stats in sorted(action_unit_stats.items()):
        if stats['total_executions'] > 5:  # Filter out rarely used action units
            action_units.append(key.replace('_', ' ').title())
            reuse_data['Direct Reuse'].append(np.mean(stats['direct_reuse_rates']) * 100)
            reuse_data['Adapted Reuse'].append(np.mean(stats['adapted_reuse_rates']) * 100)
            reuse_data['New Generation'].append(np.mean(stats['new_generation_rates']) * 100)
            time_saved_data.append(np.mean(stats['avg_time_saved']))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Stacked bar chart for reuse patterns
    x = np.arange(len(action_units))
    width = 0.6
    
    bottom = np.zeros(len(action_units))
    for reuse_type, color in [('Direct Reuse', COLOR_SCHEME['direct_reuse']),
                              ('Adapted Reuse', COLOR_SCHEME['adapted_reuse']),
                              ('New Generation', COLOR_SCHEME['new_generation'])]:
        values = reuse_data[reuse_type]
        ax1.bar(x, values, width, bottom=bottom, label=reuse_type, color=color, alpha=0.8)
        
        # Add percentage labels
        for i, (bot, val) in enumerate(zip(bottom, values)):
            if val > 5:  # Only label segments > 5%
                ax1.text(i, bot + val/2, f'{val:.0f}%', ha='center', va='center', fontsize=9)
        
        bottom += values
    
    ax1.set_xlabel('Action Unit', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Code Reuse Patterns by Action Unit', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(action_units, rotation=45, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Bar chart for average time saved
    colors = [AGENT_COLORS.get(au.split('_')[0].lower(), '#34495e') for au in action_unit_stats.keys() if action_unit_stats[au]['total_executions'] > 5]
    bars = ax2.bar(x, time_saved_data, width, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, val in zip(bars, time_saved_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Action Unit', fontsize=12)
    ax2.set_ylabel('Average Time Saved (seconds)', fontsize=12)
    ax2.set_title('Average Time Saved per Execution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(action_units, rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add agent legend
    agent_patches = [mpatches.Patch(color=color, label=agent.replace('_', ' ').title()) 
                    for agent, color in AGENT_COLORS.items()]
    ax2.legend(handles=agent_patches, loc='upper left', title='Agent Type')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_memory_evolution_plot(data: List[Dict], output_path: str):
    """Create a plot showing how memory usage evolves over problem solving"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Process usage records for visualization
    all_records = []
    for session_data in data:
        for record in session_data['usage_records']:
            all_records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_records)
    
    # Create timeline visualization
    agents = df['agent_role'].unique()
    agent_y_positions = {agent: i for i, agent in enumerate(agents)}
    
    # Plot each memory usage event
    for _, record in df.iterrows():
        y_pos = agent_y_positions[record['agent_role']]
        x_pos = record['timestamp'] - df['timestamp'].min()
        
        # Determine marker style based on reuse type
        if record['reuse_type'] == 'direct_reuse':
            marker = 'o'
            size = 100
            color = COLOR_SCHEME['direct_reuse']
        elif record['reuse_type'] == 'adapted_reuse':
            marker = 's'
            size = 80
            color = COLOR_SCHEME['adapted_reuse']
        else:  # new_generation
            marker = '^'
            size = 120
            color = COLOR_SCHEME['new_generation']
        
        # Plot with transparency based on success
        alpha = 0.8 if record['success'] else 0.3
        ax.scatter(x_pos, y_pos, s=size, marker=marker, color=color, alpha=alpha, 
                  edgecolors='black', linewidth=0.5)
    
    # Customize plot
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([agent.replace('_', ' ').title() for agent in agents])
    ax.set_xlabel('Time Since Start (seconds)', fontsize=12)
    ax.set_ylabel('Agent', fontsize=12)
    ax.set_title('Memory Usage Evolution During Problem Solving', fontsize=16, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], marker='o', s=100, color=COLOR_SCHEME['direct_reuse'], 
                   edgecolors='black', linewidth=0.5, label='Direct Reuse'),
        plt.scatter([], [], marker='s', s=80, color=COLOR_SCHEME['adapted_reuse'], 
                   edgecolors='black', linewidth=0.5, label='Adapted Reuse'),
        plt.scatter([], [], marker='^', s=120, color=COLOR_SCHEME['new_generation'], 
                   edgecolors='black', linewidth=0.5, label='New Generation'),
        plt.scatter([], [], marker='o', s=100, color='gray', alpha=0.3, 
                   edgecolors='black', linewidth=0.5, label='Failed Execution')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add task context annotations
    task_changes = df.groupby('task_context')['timestamp'].min()
    for task, timestamp in task_changes.items():
        x_pos = timestamp - df['timestamp'].min()
        ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5)
        ax.text(x_pos, len(agents) - 0.5, task, rotation=90, fontsize=9, 
               va='bottom', ha='right', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_statistics(data: List[Dict], output_path: str):
    """Create a summary statistics figure"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Aggregate statistics across all sessions
    total_uses = 0
    total_direct = 0
    total_adapted = 0
    total_new = 0
    total_time_saved = 0
    
    for session_data in data:
        stats = session_data['statistics']
        total_uses += stats['total_memory_uses']
        total_direct += stats['total_direct_reuses']
        total_adapted += stats['total_adapted_reuses']
        total_new += stats['total_new_generations']
        total_time_saved += stats['total_time_saved']
    
    # Pie chart of reuse types
    sizes = [total_direct, total_adapted, total_new]
    labels = ['Direct Reuse', 'Adapted Reuse', 'New Generation']
    colors = [COLOR_SCHEME['direct_reuse'], COLOR_SCHEME['adapted_reuse'], COLOR_SCHEME['new_generation']]
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Distribution of Memory Usage Types', fontsize=14, fontweight='bold')
    
    # Bar chart of time saved by agent
    agent_time_saved = {}
    for session_data in data:
        for record in session_data['usage_records']:
            agent = record['agent_role']
            if agent not in agent_time_saved:
                agent_time_saved[agent] = 0
            agent_time_saved[agent] += record['time_saved']
    
    agents = list(agent_time_saved.keys())
    times = list(agent_time_saved.values())
    colors = [AGENT_COLORS.get(agent.lower(), '#34495e') for agent in agents]
    
    bars = ax2.bar(agents, times, color=colors, alpha=0.8)
    for bar, time in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
                f'{time:.0f}s', ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Agent', fontsize=12)
    ax2.set_ylabel('Total Time Saved (seconds)', fontsize=12)
    ax2.set_title('Total Time Saved by Agent', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([agent.replace('_', ' ').title() for agent in agents], rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Summary metrics
    metrics = {
        'Total Memory Uses': total_uses,
        'Direct Reuse Rate': f'{(total_direct/total_uses)*100:.1f}%' if total_uses > 0 else '0%',
        'Adapted Reuse Rate': f'{(total_adapted/total_uses)*100:.1f}%' if total_uses > 0 else '0%',
        'Total Time Saved': f'{total_time_saved/60:.1f} minutes',
        'Avg Time Saved/Use': f'{total_time_saved/total_uses:.1f}s' if total_uses > 0 else '0s'
    }
    
    ax3.axis('off')
    y_pos = 0.9
    for metric, value in metrics.items():
        ax3.text(0.1, y_pos, f'{metric}:', fontsize=12, fontweight='bold')
        ax3.text(0.6, y_pos, str(value), fontsize=12)
        y_pos -= 0.15
    ax3.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    # Efficiency improvement over time (sessions)
    efficiency_by_session = []
    for i, session_data in enumerate(data):
        timeline = session_data['statistics']['reuse_efficiency_timeline']
        if timeline:
            final_efficiency = timeline[-1]['efficiency_ratio'] * 100
            efficiency_by_session.append((i+1, final_efficiency))
    
    if efficiency_by_session:
        sessions, efficiencies = zip(*efficiency_by_session)
        ax4.plot(sessions, efficiencies, marker='o', markersize=10, linewidth=2, color='#2ecc71')
        ax4.set_xlabel('Session Number', fontsize=12)
        ax4.set_ylabel('Final Efficiency (%)', fontsize=12)
        ax4.set_title('Memory Efficiency Improvement Across Sessions', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add trend line
        if len(sessions) > 1:
            z = np.polyfit(sessions, efficiencies, 1)
            p = np.poly1d(z)
            ax4.plot(sessions, p(sessions), "--", alpha=0.5, color='red', label='Trend')
            ax4.legend()
    
    plt.suptitle('GenoMAS Memory Reuse Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to generate all visualizations"""
    # Load data
    data = load_memory_data()
    
    if not data:
        print("No memory tracking data found. Please run experiments first.")
        return
    
    print(f"Loaded {len(data)} memory tracking sessions")
    
    # Create output directory
    output_dir = "./output/memory_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Creating cumulative efficiency plot...")
    create_cumulative_efficiency_plot(data, os.path.join(output_dir, "cumulative_efficiency.png"))
    
    print("Creating action unit reuse patterns...")
    create_action_unit_reuse_patterns(data, os.path.join(output_dir, "action_unit_patterns.png"))
    
    print("Creating memory evolution plot...")
    create_memory_evolution_plot(data, os.path.join(output_dir, "memory_evolution.png"))
    
    print("Creating summary statistics...")
    create_summary_statistics(data, os.path.join(output_dir, "summary_statistics.png"))
    
    print(f"\nAll visualizations saved to {output_dir}")
    
    # Generate report
    generate_report(data, output_dir)


def generate_report(data: List[Dict], output_dir: str):
    """Generate a text report summarizing the findings"""
    report_path = os.path.join(output_dir, "memory_reuse_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("GenoMAS Memory Reuse Efficiency Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sessions analyzed: {len(data)}\n\n")
        
        # Calculate overall statistics
        total_uses = sum(d['statistics']['total_memory_uses'] for d in data)
        total_direct = sum(d['statistics']['total_direct_reuses'] for d in data)
        total_adapted = sum(d['statistics']['total_adapted_reuses'] for d in data)
        total_time_saved = sum(d['statistics']['total_time_saved'] for d in data)
        
        f.write("Overall Statistics:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total memory operations: {total_uses}\n")
        f.write(f"Direct reuse rate: {(total_direct/total_uses)*100:.1f}%\n")
        f.write(f"Adapted reuse rate: {(total_adapted/total_uses)*100:.1f}%\n")
        f.write(f"Total time saved: {total_time_saved/60:.1f} minutes\n")
        f.write(f"Average time saved per operation: {total_time_saved/total_uses:.1f} seconds\n\n")
        
        # Most efficient action units
        f.write("Most Efficient Action Units:\n")
        f.write("-" * 30 + "\n")
        
        action_unit_efficiency = {}
        for session_data in data:
            for key, stats in session_data['statistics']['reuse_patterns'].items():
                if key not in action_unit_efficiency:
                    action_unit_efficiency[key] = []
                efficiency = stats['direct_reuse_rate'] + stats['adapted_reuse_rate']
                action_unit_efficiency[key].append(efficiency)
        
        sorted_units = sorted(action_unit_efficiency.items(), 
                             key=lambda x: np.mean(x[1]), reverse=True)
        
        for unit, efficiencies in sorted_units[:5]:
            avg_efficiency = np.mean(efficiencies) * 100
            f.write(f"{unit}: {avg_efficiency:.1f}% reuse rate\n")
        
        f.write(f"\nReport saved to: {report_path}\n")
    
    print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()