import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from collections import defaultdict
import pandas as pd


class CollaborationVisualizer:
    """Visualize agent collaboration patterns from collected data"""
    
    def __init__(self, data_dir: str = "./output/collaboration_data"):
        self.data_dir = Path(data_dir)
        self.agent_colors = {
            "PI": "#FF6B6B",           # Red
            "GEO Agent": "#4ECDC4",    # Teal
            "TCGA Agent": "#45B7D1",   # Blue
            "Statistician Agent": "#96CEB4",  # Green
            "Code Reviewer": "#DDA0DD",  # Plum
            "Domain Expert": "#FECA57"   # Yellow
        }
        
        # Professional color scheme
        self.figsize = (16, 10)
        self.font_family = "Arial"
        plt.rcParams['font.family'] = self.font_family
        plt.rcParams['font.size'] = 10
        
    def load_session_data(self, session_file: str) -> dict:
        """Load collaboration data from a session file"""
        filepath = self.data_dir / session_file
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def create_network_graph(self, session_data: dict, output_file: str):
        """Create a network graph showing agent interactions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Left panel: Static network structure
        self._draw_static_network(ax1, session_data)
        
        # Right panel: Dynamic interaction heatmap
        self._draw_interaction_heatmap(ax2, session_data)
        
        # Overall title
        fig.suptitle("GenoMAS Agent Collaboration Network", fontsize=18, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
    def _draw_static_network(self, ax, session_data):
        """Draw the static network structure with interaction counts"""
        G = nx.DiGraph()
        
        # Extract interaction data
        interaction_counts = session_data['statistics']['interaction_counts']
        
        # Add nodes and edges
        agents = set()
        for interaction_key, count in interaction_counts.items():
            sender, receiver = interaction_key.split('->')
            agents.add(sender)
            agents.add(receiver)
            if count > 0:
                G.add_edge(sender, receiver, weight=count)
        
        # Calculate node positions using spring layout with custom parameters
        pos = self._get_hierarchical_layout(list(agents))
        
        # Draw nodes
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            activity = session_data['statistics']['agent_activity'].get(node, 0)
            node_sizes.append(1000 + activity * 50)  # Scale by activity
            node_colors.append(self.agent_colors.get(node, '#999999'))
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                               node_size=node_sizes, alpha=0.9, ax=ax)
        
        # Draw edges with varying thickness based on interaction count
        edges = G.edges()
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(edge_weights) if edge_weights else 1
        
        # Normalize edge weights for visualization
        edge_widths = [1 + (w / max_weight) * 5 for w in edge_weights]
        edge_alphas = [0.3 + (w / max_weight) * 0.6 for w in edge_weights]
        
        # Draw edges with arrows
        for (u, v), width, alpha in zip(edges, edge_widths, edge_alphas):
            nx.draw_networkx_edges(G, pos, [(u, v)], width=width, alpha=alpha,
                                   edge_color='#666666', arrows=True,
                                   arrowsize=15, arrowstyle='->', ax=ax,
                                   connectionstyle="arc3,rad=0.1")
        
        # Draw labels
        labels = {node: node.replace(' Agent', '\nAgent') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, 
                                font_weight='bold', ax=ax)
        
        # Add interaction count labels on edges
        edge_labels = {(u, v): str(G[u][v]['weight']) 
                       for u, v in G.edges() if G[u][v]['weight'] > 1}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.set_title("Agent Communication Network", fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend for node sizes
        legend_elements = [
            plt.scatter([], [], s=1000, c='gray', alpha=0.6, label='Low activity'),
            plt.scatter([], [], s=2000, c='gray', alpha=0.6, label='Medium activity'),
            plt.scatter([], [], s=3000, c='gray', alpha=0.6, label='High activity')
        ]
        ax.legend(handles=legend_elements, loc='upper left', title='Agent Activity')
        
    def _get_hierarchical_layout(self, agents: List[str]) -> Dict[str, Tuple[float, float]]:
        """Create a hierarchical layout for the agents"""
        pos = {}
        
        # Define hierarchy levels
        levels = {
            "PI": 0,
            "GEO Agent": 1,
            "TCGA Agent": 1,
            "Statistician Agent": 1,
            "Code Reviewer": 2,
            "Domain Expert": 2
        }
        
        # Group agents by level
        level_agents = defaultdict(list)
        for agent in agents:
            level = levels.get(agent, 1)
            level_agents[level].append(agent)
        
        # Position agents
        y_spacing = 1.0
        for level, agents_in_level in level_agents.items():
            y = -level * y_spacing
            x_spacing = 2.0 / (len(agents_in_level) + 1)
            for i, agent in enumerate(agents_in_level):
                x = -1.0 + (i + 1) * x_spacing
                pos[agent] = (x, y)
        
        return pos
    
    def _draw_interaction_heatmap(self, ax, session_data):
        """Draw a heatmap of message types between agents"""
        interactions = session_data['interactions']
        
        # Create a matrix of message type counts
        agents = list(self.agent_colors.keys())
        message_types = set()
        interaction_matrix = defaultdict(lambda: defaultdict(int))
        
        for interaction in interactions:
            sender = interaction['sender']
            receiver = interaction['receiver']
            msg_type = interaction['message_type'].replace('_', ' ').title()
            message_types.add(msg_type)
            interaction_matrix[(sender, receiver)][msg_type] += 1
        
        # Create heatmap data
        message_types = sorted(list(message_types))
        heatmap_data = []
        labels = []
        
        for sender in agents:
            for receiver in agents:
                if sender != receiver:
                    row = []
                    for msg_type in message_types:
                        count = interaction_matrix[(sender, receiver)][msg_type]
                        row.append(count)
                    if sum(row) > 0:  # Only include pairs with interactions
                        heatmap_data.append(row)
                        labels.append(f"{sender} → {receiver}")
        
        if not heatmap_data:
            ax.text(0.5, 0.5, "No interactions recorded", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Message Type Distribution", fontsize=14, fontweight='bold')
            return
        
        # Create heatmap
        heatmap_array = np.array(heatmap_data)
        
        # Use a better colormap
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        im = ax.imshow(heatmap_array, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(message_types)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(message_types, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Message Count', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(message_types)):
                value = heatmap_array[i, j]
                if value > 0:
                    text_color = 'white' if value > heatmap_array.max() / 2 else 'black'
                    ax.text(j, i, str(int(value)), ha='center', va='center', 
                            color=text_color, fontsize=8)
        
        ax.set_title("Message Type Distribution", fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Message Type", fontweight='bold')
        ax.set_ylabel("Agent Pair", fontweight='bold')
        
    def create_temporal_flow_diagram(self, session_data: dict, output_file: str):
        """Create a temporal flow diagram showing interaction patterns over time"""
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))
        
        interactions = session_data['interactions']
        if not interactions:
            ax.text(0.5, 0.5, "No interactions recorded", 
                    ha='center', va='center', transform=ax.transAxes)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Group interactions by task
        task_interactions = defaultdict(list)
        for interaction in interactions:
            task = interaction['task_context']
            task_interactions[task].append(interaction)
        
        # Create timeline for each task
        y_offset = 0
        task_height = 10
        task_spacing = 2
        
        for task, task_ints in task_interactions.items():
            # Draw task background
            task_start = min(i['timestamp'] for i in task_ints) - session_data['start_time']
            task_end = max(i['timestamp'] for i in task_ints) - session_data['start_time']
            
            rect = FancyBboxPatch((task_start, y_offset), task_end - task_start, task_height,
                                   boxstyle="round,pad=0.1", facecolor='#f0f0f0',
                                   edgecolor='#cccccc', linewidth=1)
            ax.add_patch(rect)
            
            # Add task label
            ax.text(task_start - 5, y_offset + task_height/2, task, 
                    ha='right', va='center', fontweight='bold', fontsize=10)
            
            # Draw interactions within task
            agent_y_positions = {
                "PI": y_offset + 8,
                "GEO Agent": y_offset + 6.5,
                "TCGA Agent": y_offset + 5,
                "Statistician Agent": y_offset + 3.5,
                "Code Reviewer": y_offset + 2,
                "Domain Expert": y_offset + 0.5
            }
            
            for interaction in task_ints:
                time = interaction['timestamp'] - session_data['start_time']
                sender = interaction['sender']
                receiver = interaction['receiver']
                msg_type = interaction['message_type']
                
                if sender in agent_y_positions and receiver in agent_y_positions:
                    y1 = agent_y_positions[sender]
                    y2 = agent_y_positions[receiver]
                    
                    # Draw arrow
                    arrow_color = self.agent_colors.get(sender, '#999999')
                    ax.annotate('', xy=(time, y2), xytext=(time, y1),
                                arrowprops=dict(arrowstyle='->', color=arrow_color,
                                                alpha=0.6, lw=1.5))
                    
                    # Add message type label for important messages
                    if 'request' in msg_type or 'response' in msg_type:
                        ax.text(time + 0.5, (y1 + y2) / 2, 
                                msg_type.replace('_', ' ').title()[:12] + '...',
                                fontsize=6, rotation=45, alpha=0.7)
            
            y_offset += task_height + task_spacing
        
        # Formatting
        ax.set_xlabel("Time (seconds)", fontweight='bold', fontsize=12)
        ax.set_ylabel("Tasks and Agents", fontweight='bold', fontsize=12)
        ax.set_title("Temporal Flow of Agent Interactions", fontsize=16, fontweight='bold', pad=20)
        
        # Set limits
        ax.set_xlim(-10, max(i['timestamp'] - session_data['start_time'] for i in interactions) + 10)
        ax.set_ylim(-2, y_offset + 2)
        
        # Remove y-axis ticks
        ax.set_yticks([])
        
        # Grid
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_summary_statistics(self, session_data: dict, output_file: str):
        """Create a summary statistics visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        stats = session_data['statistics']
        
        # 1. Agent activity pie chart
        agent_activity = stats['agent_activity']
        if agent_activity:
            agents = list(agent_activity.keys())
            activities = list(agent_activity.values())
            colors = [self.agent_colors.get(agent, '#999999') for agent in agents]
            
            ax1.pie(activities, labels=agents, colors=colors, autopct='%1.1f%%',
                    startangle=90, textprops={'fontsize': 10})
            ax1.set_title("Agent Activity Distribution", fontweight='bold', fontsize=12)
        
        # 2. Message type distribution
        msg_types = stats['message_type_counts']
        if msg_types:
            msg_labels = [mt.replace('_', ' ').title() for mt in msg_types.keys()]
            msg_counts = list(msg_types.values())
            
            bars = ax2.bar(range(len(msg_labels)), msg_counts, color='#4ECDC4')
            ax2.set_xticks(range(len(msg_labels)))
            ax2.set_xticklabels(msg_labels, rotation=45, ha='right')
            ax2.set_ylabel("Count", fontweight='bold')
            ax2.set_title("Message Type Frequency", fontweight='bold', fontsize=12)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # 3. Top interaction pairs
        interaction_counts = stats['interaction_counts']
        if interaction_counts:
            sorted_pairs = sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            pair_labels = [pair[0] for pair in sorted_pairs]
            pair_counts = [pair[1] for pair in sorted_pairs]
            
            bars = ax3.barh(range(len(pair_labels)), pair_counts, color='#96CEB4')
            ax3.set_yticks(range(len(pair_labels)))
            ax3.set_yticklabels(pair_labels, fontsize=9)
            ax3.set_xlabel("Interaction Count", fontweight='bold')
            ax3.set_title("Top 10 Agent Interaction Pairs", fontweight='bold', fontsize=12)
            
            # Add value labels
            for i, (label, count) in enumerate(sorted_pairs):
                ax3.text(count + 0.5, i, str(count), va='center', fontsize=8)
        
        # 4. Session summary text
        ax4.axis('off')
        summary_text = f"""Session Summary
        
Total Interactions: {stats['total_interactions']}
Session Duration: {session_data['duration']:.2f} seconds
Tasks Processed: {len(stats['tasks_processed'])}
Active Agents: {len(stats['agent_activity'])}

Most Active Agent: {max(stats['agent_activity'].items(), key=lambda x: x[1])[0] if stats['agent_activity'] else 'N/A'}
Most Common Message: {max(stats['message_type_counts'].items(), key=lambda x: x[1])[0].replace('_', ' ').title() if stats['message_type_counts'] else 'N/A'}
"""
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.5))
        
        fig.suptitle("GenoMAS Collaboration Summary Statistics", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def main():
    """Main function to generate visualizations"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_collaboration.py <session_file>")
        print("Example: python visualize_collaboration.py genomas_run_v1_20241224_143022.json")
        sys.exit(1)
    
    session_file = sys.argv[1]
    output_dir = "./output/collaboration_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = CollaborationVisualizer()
    
    # Load session data
    try:
        session_data = visualizer.load_session_data(session_file)
        print(f"Loaded session data from {session_file}")
    except FileNotFoundError:
        print(f"Error: Could not find session file {session_file}")
        sys.exit(1)
    
    # Generate visualizations
    base_name = session_file.replace('.json', '')
    
    # 1. Network graph
    network_output = os.path.join(output_dir, f"{base_name}_network.png")
    visualizer.create_network_graph(session_data, network_output)
    print(f"Created network graph: {network_output}")
    
    # 2. Temporal flow diagram
    temporal_output = os.path.join(output_dir, f"{base_name}_temporal.png")
    visualizer.create_temporal_flow_diagram(session_data, temporal_output)
    print(f"Created temporal flow diagram: {temporal_output}")
    
    # 3. Summary statistics
    summary_output = os.path.join(output_dir, f"{base_name}_summary.png")
    visualizer.create_summary_statistics(session_data, summary_output)
    print(f"Created summary statistics: {summary_output}")
    
    print("\nAll visualizations created successfully!")


if __name__ == "__main__":
    main()