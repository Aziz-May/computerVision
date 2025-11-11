import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import pickle

def load_stips(filename='stips_data.pkl'):
    """Load STIPs from pickle file"""
    print(f"📂 Loading STIPs from {filename}...")
    with open(filename, 'rb') as f:
        stips = pickle.load(f)
    print(f"✅ Loaded {len(stips)} STIPs")
    return stips

def create_cylindrical_coordinates(stips):
    """
    Convert STIPs to cylindrical coordinates (r, theta, z)
    where z is the time axis
    """
    xs = np.array([s[0] for s in stips])
    ys = np.array([s[1] for s in stips])
    ts = np.array([s[2] for s in stips])
    responses = np.array([s[3] for s in stips])
    
    # Find center in x-y plane
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    print(f"\n📍 Center (X, Y): ({cx:.1f}, {cy:.1f})")
    
    # Convert to cylindrical coordinates
    # r = distance from center axis
    r = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    # theta = angle around axis
    theta = np.arctan2(ys - cy, xs - cx)
    # z = time (unchanged)
    z = ts
    
    max_r = np.max(r)
    max_z = np.max(z)
    
    print(f"📏 Cylinder dimensions:")
    print(f"   Max radius: {max_r:.1f}")
    print(f"   Height (time): {max_z:.1f}")
    
    return r, theta, z, responses, cx, cy, max_r, max_z

def visualize_cylinder_with_points(stips, view='3d'):
    """
    Visualize the cylindrical volume with STIP points
    """
    r, theta, z, responses, cx, cy, max_r, max_z = create_cylindrical_coordinates(stips)
    
    # Convert back to Cartesian for visualization
    xs = np.array([s[0] for s in stips])
    ys = np.array([s[1] for s in stips])
    ts = np.array([s[2] for s in stips])
    
    # Normalize responses for color
    responses_norm = (responses - responses.min()) / (responses.max() - responses.min()) if responses.max() > 0 else responses
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # ============ VIEW 1: 3D with cylinder wireframe ============
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor('black')
    
    # Draw cylinder wireframe
    theta_cyl = np.linspace(0, 2*np.pi, 50)
    z_cyl = np.linspace(0, max_z, 20)
    Theta_cyl, Z_cyl = np.meshgrid(theta_cyl, z_cyl)
    X_cyl = cx + max_r * np.cos(Theta_cyl)
    Y_cyl = cy + max_r * np.sin(Theta_cyl)
    
    ax1.plot_wireframe(X_cyl, Y_cyl, Z_cyl, color='cyan', alpha=0.2, linewidth=0.5)
    
    # Plot STIP points
    scatter1 = ax1.scatter(xs, ys, ts, c=responses_norm, cmap='hot', s=5, alpha=0.6)
    
    ax1.set_xlabel('X (spatial)', fontsize=10, color='white')
    ax1.set_ylabel('Y (spatial)', fontsize=10, color='white')
    ax1.set_zlabel('Time (frames)', fontsize=10, color='white')
    ax1.set_title('3D View: STIPs in Cylinder', fontsize=12, fontweight='bold', color='white')
    ax1.tick_params(colors='white')
    plt.colorbar(scatter1, ax=ax1, label='Response', pad=0.1, shrink=0.6)
    
    # ============ VIEW 2: Cylindrical coordinates (unwrapped) ============
    ax2 = fig.add_subplot(132)
    
    scatter2 = ax2.scatter(theta * 180/np.pi, z, c=r, cmap='viridis', s=10, alpha=0.6)
    ax2.set_xlabel('Angle θ (degrees)', fontsize=11)
    ax2.set_ylabel('Time (frames)', fontsize=11)
    ax2.set_title('Unwrapped Cylinder View\n(Angle vs Time, colored by Radius)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Radius from center')
    
    # ============ VIEW 3: Top-down view (X-Y plane) ============
    ax3 = fig.add_subplot(133)
    
    # Draw circle boundary
    circle_theta = np.linspace(0, 2*np.pi, 100)
    circle_x = cx + max_r * np.cos(circle_theta)
    circle_y = cy + max_r * np.sin(circle_theta)
    ax3.plot(circle_x, circle_y, 'c--', linewidth=2, alpha=0.5, label='Cylinder boundary')
    
    scatter3 = ax3.scatter(xs, ys, c=ts, cmap='plasma', s=15, alpha=0.7)
    ax3.plot(cx, cy, 'r+', markersize=15, markeredgewidth=2, label='Center')
    
    ax3.set_xlabel('X (spatial)', fontsize=11)
    ax3.set_ylabel('Y (spatial)', fontsize=11)
    ax3.set_title('Top-Down View (X-Y plane)\nColored by Time', 
                  fontsize=12, fontweight='bold')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    plt.colorbar(scatter3, ax=ax3, label='Time (frames)')
    
    plt.tight_layout()
    plt.savefig('cylinder_views.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("\n💾 Saved as 'cylinder_views.png'")
    plt.show()

def visualize_rotating_cylinder(stips):
    """
    Create rotating 3D visualization of cylinder with points
    """
    from matplotlib.animation import FuncAnimation
    
    r, theta, z, responses, cx, cy, max_r, max_z = create_cylindrical_coordinates(stips)
    
    xs = np.array([s[0] for s in stips])
    ys = np.array([s[1] for s in stips])
    ts = np.array([s[2] for s in stips])
    
    responses_norm = (responses - responses.min()) / (responses.max() - responses.min()) if responses.max() > 0 else responses
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Draw cylinder
    theta_cyl = np.linspace(0, 2*np.pi, 30)
    z_cyl = np.linspace(0, max_z, 15)
    Theta_cyl, Z_cyl = np.meshgrid(theta_cyl, z_cyl)
    X_cyl = cx + max_r * np.cos(Theta_cyl)
    Y_cyl = cy + max_r * np.sin(Theta_cyl)
    
    def update(frame):
        ax.clear()
        ax.set_facecolor('black')
        
        # Draw cylinder wireframe
        ax.plot_wireframe(X_cyl, Y_cyl, Z_cyl, color='cyan', alpha=0.15, linewidth=0.5)
        
        # Draw top and bottom circles
        circle_theta = np.linspace(0, 2*np.pi, 50)
        for z_level in [0, max_z]:
            circle_x = cx + max_r * np.cos(circle_theta)
            circle_y = cy + max_r * np.sin(circle_theta)
            ax.plot(circle_x, circle_y, z_level, 'c-', linewidth=2, alpha=0.3)
        
        # Plot points
        ax.scatter(xs, ys, ts, c=responses_norm, cmap='hot', s=8, alpha=0.7)
        
        ax.set_xlabel('X', fontsize=10, color='white')
        ax.set_ylabel('Y', fontsize=10, color='white')
        ax.set_zlabel('Time', fontsize=10, color='white')
        ax.set_title(f'Cylindrical STIP Cloud (Rotating)\nRadius={max_r:.1f}, Height={max_z:.1f}', 
                     fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        ax.view_init(elev=20, azim=frame)
        
        ax.set_xlim([cx - max_r - 20, cx + max_r + 20])
        ax.set_ylim([cy - max_r - 20, cy + max_r + 20])
        ax.set_zlim([0, max_z])
        
        return ax,
    
    print("\n🔄 Creating rotating animation...")
    anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)
    
    try:
        anim.save('cylinder_rotating.gif', writer='pillow', fps=20)
        print("💾 Saved rotating animation as 'cylinder_rotating.gif'")
    except Exception as e:
        print(f"⚠️ Could not save animation: {e}")
    
    plt.show()

def visualize_cylindrical_slices(stips, n_slices=6):
    """
    Show time slices through the cylinder
    """
    r, theta, z, responses, cx, cy, max_r, max_z = create_cylindrical_coordinates(stips)
    
    xs = np.array([s[0] for s in stips])
    ys = np.array([s[1] for s in stips])
    ts = np.array([s[2] for s in stips])
    
    # Create time slices
    time_slices = np.linspace(0, max_z, n_slices + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i >= n_slices:
            break
        
        t_start = time_slices[i]
        t_end = time_slices[i + 1]
        
        # Select points in this time range
        mask = (ts >= t_start) & (ts < t_end)
        xs_slice = xs[mask]
        ys_slice = ys[mask]
        
        # Draw circle
        circle_theta = np.linspace(0, 2*np.pi, 100)
        circle_x = cx + max_r * np.cos(circle_theta)
        circle_y = cy + max_r * np.sin(circle_theta)
        ax.plot(circle_x, circle_y, 'c--', linewidth=2, alpha=0.5)
        
        # Plot points
        if len(xs_slice) > 0:
            ax.scatter(xs_slice, ys_slice, c='red', s=20, alpha=0.6)
        
        ax.plot(cx, cy, 'b+', markersize=15, markeredgewidth=2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Time Slice: {t_start:.0f}-{t_end:.0f}\n({len(xs_slice)} points)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([cx - max_r - 20, cx + max_r + 20])
        ax.set_ylim([cy - max_r - 20, cy + max_r + 20])
    
    plt.tight_layout()
    plt.savefig('cylinder_time_slices.png', dpi=150, bbox_inches='tight')
    print("💾 Saved time slices as 'cylinder_time_slices.png'")
    plt.show()

def main():
    """Main execution"""
    print("="*70)
    print("3D CYLINDRICAL VOLUME VISUALIZATION WITH STIP CLOUD")
    print("="*70)
    
    # Load STIPs
    stips = load_stips('stips_data.pkl')
    
    # Create all visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    print("\n1️⃣ Creating multi-view visualization...")
    visualize_cylinder_with_points(stips)
    
    print("\n2️⃣ Creating time slice views...")
    visualize_cylindrical_slices(stips, n_slices=6)
    
    print("\n3️⃣ Creating rotating 3D animation...")
    visualize_rotating_cylinder(stips)
    
    print("\n" + "="*70)
    print("✨ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - cylinder_views.png (3 different views)")
    print("   - cylinder_time_slices.png (temporal slices)")
    print("   - cylinder_rotating.gif (animated rotation)")

if __name__ == "__main__":
    main()