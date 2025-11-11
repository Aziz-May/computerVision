import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os

class CylindricalTraceTransform:
    """
    Implementation of 3D Cylindrical Trace Transform (CTT)
    as described in Goudelis et al. paper
    """
    
    def __init__(self, theta_step=9, p_bins=100, phi_bins=180):
        """
        Initialize CTT parameters
        
        Args:
            theta_step: Rotation angle step in degrees (default 9°)
            p_bins: Number of bins for distance parameter p
            phi_bins: Number of bins for angle parameter phi
        """
        self.theta_step = theta_step
        self.p_bins = p_bins
        self.phi_bins = phi_bins
        self.theta_angles = np.arange(0, 180, theta_step)
        
    def create_3d_volume(self, stips, volume_shape=None):
        """
        Create a 3D binary volume from STIP points
        
        Args:
            stips: List of (x, y, t, response) tuples
            volume_shape: Tuple (width, height, depth), if None auto-calculate
            
        Returns:
            volume: 3D numpy array with STIP points
            center: Center of mass coordinates
        """
        if len(stips) == 0:
            raise ValueError("No STIPs provided")
        
        # Extract coordinates
        xs = np.array([s[0] for s in stips])
        ys = np.array([s[1] for s in stips])
        ts = np.array([s[2] for s in stips])
        responses = np.array([s[3] for s in stips])
        
        # Determine volume dimensions
        if volume_shape is None:
            w = int(xs.max()) + 1
            h = int(ys.max()) + 1
            d = int(ts.max()) + 1
        else:
            w, h, d = volume_shape
        
        # Create binary volume
        volume = np.zeros((h, w, d), dtype=np.float32)
        
        # Fill volume with STIP points (weighted by response)
        for x, y, t, resp in zip(xs, ys, ts, responses):
            xi, yi, ti = int(x), int(y), int(t)
            if 0 <= xi < w and 0 <= yi < h and 0 <= ti < d:
                volume[yi, xi, ti] = 1.0  # Binary representation
        
        # Calculate center of mass
        nonzero = np.argwhere(volume > 0)
        if len(nonzero) > 0:
            center = nonzero.mean(axis=0)
        else:
            center = np.array([h/2, w/2, d/2])
        
        return volume, center
    
    def normalize_volume(self, volume, center):
        """
        Normalize volume to be scale and translation invariant
        
        Args:
            volume: 3D binary volume
            center: Center of mass
            
        Returns:
            normalized_volume: Scale-normalized volume
            scale_factor: Scaling factor used
        """
        # Find maximum distance from center to any point
        nonzero_points = np.argwhere(volume > 0)
        if len(nonzero_points) == 0:
            return volume, 1.0
        
        distances = np.linalg.norm(nonzero_points - center, axis=1)
        d_max = distances.max()
        
        if d_max == 0:
            return volume, 1.0
        
        # Scale so that d_max = 1
        scale_factor = 1.0 / d_max
        
        return volume, scale_factor
    
    def compute_trace_transform_2d(self, projection, functional='sum'):
        """
        Compute 2D Trace Transform on a planar projection
        
        Args:
            projection: 2D numpy array
            functional: Type of functional to apply ('sum', 'mean', 'median', 'max', 'std')
            
        Returns:
            trace_transform: 2D array of shape (p_bins, phi_bins)
        """
        h, w = projection.shape
        center_x, center_y = w / 2, h / 2
        
        # Maximum distance from center
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        # Initialize trace transform
        trace = np.zeros((self.p_bins, self.phi_bins))
        
        # Iterate over angles phi
        for phi_idx, phi in enumerate(np.linspace(0, np.pi, self.phi_bins)):
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            # Iterate over distances p
            for p_idx, p in enumerate(np.linspace(-max_dist, max_dist, self.p_bins)):
                # Collect points along line: p = x*cos(phi) + y*sin(phi)
                line_values = []
                
                # Sample along perpendicular direction
                for t in np.linspace(-max_dist, max_dist, 200):
                    # Point on line: (x, y) = (p*cos(phi) - t*sin(phi), p*sin(phi) + t*cos(phi))
                    x = center_x + p * cos_phi - t * sin_phi
                    y = center_y + p * sin_phi + t * cos_phi
                    
                    # Check bounds and interpolate
                    if 0 <= x < w-1 and 0 <= y < h-1:
                        # Bilinear interpolation
                        x0, y0 = int(x), int(y)
                        x1, y1 = x0 + 1, y0 + 1
                        
                        dx, dy = x - x0, y - y0
                        
                        val = (projection[y0, x0] * (1-dx) * (1-dy) +
                               projection[y0, x1] * dx * (1-dy) +
                               projection[y1, x0] * (1-dx) * dy +
                               projection[y1, x1] * dx * dy)
                        
                        if val > 0:
                            line_values.append(val)
                
                # Apply functional
                if len(line_values) > 0:
                    if functional == 'sum':
                        trace[p_idx, phi_idx] = np.sum(line_values)
                    elif functional == 'mean':
                        trace[p_idx, phi_idx] = np.mean(line_values)
                    elif functional == 'median':
                        trace[p_idx, phi_idx] = np.median(line_values)
                    elif functional == 'max':
                        trace[p_idx, phi_idx] = np.max(line_values)
                    elif functional == 'std':
                        trace[p_idx, phi_idx] = np.std(line_values)
        
        return trace
    
    def compute_cylindrical_trace_transform(self, volume, center, functional='sum'):
        """
        Compute 3D Cylindrical Trace Transform
        
        Args:
            volume: 3D binary volume
            center: Center of mass
            functional: Functional to use for trace calculation
            
        Returns:
            ctt: 2D Cylindrical Trace Transform
        """
        h, w, d = volume.shape
        cy, cx, cz = center
        
        # Initialize accumulated transform
        ctt = np.zeros((self.p_bins, self.phi_bins))
        
        print(f"Computing CTT with {len(self.theta_angles)} rotation angles...")
        
        # Rotate around z-axis (time axis)
        for theta_idx, theta in enumerate(self.theta_angles):
            theta_rad = np.deg2rad(theta)
            
            # Create rotation matrix around z-axis
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)
            
            # Create 2D projection by rotating and summing along rotated axis
            projection = np.zeros((h, w))
            
            # For each point in volume, rotate and project
            for z in range(d):
                for y in range(h):
                    for x in range(w):
                        if volume[y, x, z] > 0:
                            # Translate to origin
                            x_c = x - cx
                            y_c = y - cy
                            
                            # Rotate around z-axis
                            x_rot = x_c * cos_t - y_c * sin_t
                            y_rot = x_c * sin_t + y_c * cos_t
                            
                            # Translate back
                            x_new = int(x_rot + cx)
                            y_new = int(y_rot + cy)
                            
                            # Accumulate in projection
                            if 0 <= x_new < w and 0 <= y_new < h:
                                projection[y_new, x_new] += volume[y, x, z]
            
            # Compute trace transform on this projection
            trace = self.compute_trace_transform_2d(projection, functional)
            
            # Accumulate
            ctt += trace
            
            if (theta_idx + 1) % 5 == 0:
                print(f"  Processed {theta_idx + 1}/{len(self.theta_angles)} angles")
        
        return ctt
    
    def extract_volumetric_triple_features(self, ctt, diametric_func='mean', circus_func='std'):
        """
        Extract Volumetric Triple Features from CTT
        
        Args:
            ctt: Cylindrical Trace Transform
            diametric_func: Functional for diametric operation ('mean', 'std', 'max', 'median')
            circus_func: Functional for circus operation ('mean', 'std', 'max', 'median')
            
        Returns:
            triple_feature: Single scalar feature value
        """
        # Step 1: Apply diametric functional along columns (p direction)
        if diametric_func == 'mean':
            circus = np.mean(ctt, axis=0)
        elif diametric_func == 'std':
            circus = np.std(ctt, axis=0)
        elif diametric_func == 'max':
            circus = np.max(ctt, axis=0)
        elif diametric_func == 'median':
            circus = np.median(ctt, axis=0)
        
        # Step 2: Apply circus functional
        if circus_func == 'mean':
            feature = np.mean(circus)
        elif circus_func == 'std':
            feature = np.std(circus)
        elif circus_func == 'max':
            feature = np.max(circus)
        elif circus_func == 'median':
            feature = np.median(circus)
        
        return feature
    
    def compute_feature_vector(self, stips, functionals_list=None):
        """
        Compute complete VTF feature vector
        
        Args:
            stips: List of STIP points
            functionals_list: List of (trace_func, diam_func, circ_func) tuples
            
        Returns:
            feature_vector: Complete feature vector with ratios
        """
        if functionals_list is None:
            # Default functionals
            functionals_list = [
                ('sum', 'mean', 'std'),
                ('sum', 'std', 'mean'),
                ('mean', 'mean', 'std'),
                ('max', 'max', 'std'),
                ('std', 'mean', 'mean'),
            ]
        
        # Create and normalize volume
        print("Creating 3D volume from STIPs...")
        volume, center = self.create_3d_volume(stips)
        volume, scale = self.normalize_volume(volume, center)
        
        # Compute CTTs with different functionals
        triple_features = []
        
        for trace_f, diam_f, circ_f in functionals_list:
            print(f"\nComputing CTT with functional: {trace_f}")
            ctt = self.compute_cylindrical_trace_transform(volume, center, trace_f)
            
            print(f"Extracting triple feature ({diam_f}, {circ_f})...")
            tf = self.extract_volumetric_triple_features(ctt, diam_f, circ_f)
            triple_features.append(tf)
        
        # Compute ratios between all pairs
        feature_vector = []
        for i in range(len(triple_features)):
            for j in range(len(triple_features)):
                if i != j and triple_features[j] != 0:
                    ratio = triple_features[i] / triple_features[j]
                    feature_vector.append(ratio)
        
        return np.array(feature_vector), triple_features

    def visualize_ctt(self, ctt, title="Cylindrical Trace Transform"):
        """
        Visualize the CTT as a 2D heatmap
        """
        plt.figure(figsize=(12, 6))
        plt.imshow(ctt, aspect='auto', cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Transform Value')
        plt.xlabel('φ (angle) bins')
        plt.ylabel('ρ (distance) bins')
        plt.title(title)
        plt.tight_layout()
        plt.savefig('ctt_visualization.png', dpi=150, bbox_inches='tight')
        print("💾 Saved CTT visualization as 'ctt_visualization.png'")
        plt.show()


def load_stips_from_file(filepath):
    """
    Load STIPs from a pickle file or text file
    """
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        # Assume text format: x y t response
        stips = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    stips.append((float(parts[0]), float(parts[1]), 
                                float(parts[2]), float(parts[3])))
        return stips


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("3D CYLINDRICAL TRACE TRANSFORM")
    print("="*60)
    
    # Option 1: Load STIPs from file
    # stips = load_stips_from_file('stips_data.pkl')
    
    # Option 2: Generate synthetic STIPs for testing
    print("\n📊 Generating synthetic STIP data for demonstration...")
    np.random.seed(42)
    n_points = 500
    
    # Simulate a simple action trajectory
    t_vals = np.linspace(0, 100, n_points)
    x_vals = 50 + 30 * np.sin(t_vals / 10) + np.random.randn(n_points) * 2
    y_vals = 50 + 30 * np.cos(t_vals / 10) + np.random.randn(n_points) * 2
    responses = np.abs(np.random.randn(n_points))
    
    stips = [(x, y, t, r) for x, y, t, r in zip(x_vals, y_vals, t_vals, responses)]
    
    print(f"✅ Generated {len(stips)} synthetic STIPs")
    
    # Initialize CTT
    ctt_extractor = CylindricalTraceTransform(theta_step=9, p_bins=80, phi_bins=120)
    
    # Compute single CTT for visualization
    print("\n" + "="*60)
    print("Computing CTT for visualization...")
    print("="*60)
    
    volume, center = ctt_extractor.create_3d_volume(stips)
    volume, scale = ctt_extractor.normalize_volume(volume, center)
    
    ctt = ctt_extractor.compute_cylindrical_trace_transform(volume, center, functional='sum')
    
    # Visualize
    ctt_extractor.visualize_ctt(ctt, "3D Cylindrical Trace Transform (Sum functional)")
    
    # Compute complete feature vector
    print("\n" + "="*60)
    print("Computing Volumetric Triple Features...")
    print("="*60)
    
    feature_vector, triple_features = ctt_extractor.compute_feature_vector(stips)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"   Triple features: {triple_features}")
    print(f"   Feature vector length: {len(feature_vector)}")
    print(f"   Feature vector sample: {feature_vector[:5]}")
    
    # Save results
    np.save('feature_vector.npy', feature_vector)
    print("\n💾 Saved feature vector to 'feature_vector.npy'")