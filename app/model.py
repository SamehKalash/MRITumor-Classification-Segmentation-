"""
ViT Model Architecture
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, LayerNormalization, MultiHeadAttention
import numpy as np
import cv2

# Model Hyperparameters (matching notebook exactly)
IMAGE_SIZE = 256
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256
HIDDEN_SIZE = 64  # Hidden_size in notebook
NUM_HEADS = 4     # Num_heads in notebook
NUM_LAYERS = 15   # num_layers in notebook
MLP_HEAD_UNITS = 1024  # mlp_head_units in notebook
NUM_CLASSES = 4

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


# Classes match EXACT structure from braintumor-vit.ipynb
class PatchEncoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.projection = Dense(HIDDEN_SIZE)
        self.position_embed = layers.Embedding(NUM_PATCHES, HIDDEN_SIZE)

    def call(self, images):
        batch = tf.shape(images)[0]
        # Extract patches
        patches = tf.image.extract_patches(
            images, sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
            rates=[1, 1, 1, 1], padding='VALID'
        )
        patches = tf.reshape(patches, [batch, -1, patches.shape[-1]])
        # Project + add position embeddings
        positions = tf.range(NUM_PATCHES)
        return self.projection(patches) + self.position_embed(positions)


class TransformerEncoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.attn = MultiHeadAttention(NUM_HEADS, HIDDEN_SIZE)
        self.mlp = tf.keras.Sequential([
            Dense(HIDDEN_SIZE * 2, activation='gelu'),
            Dense(HIDDEN_SIZE)
        ])

    def call(self, x, return_attention=False):
        # Self-attention with residual
        normed = self.norm1(x)
        if return_attention:
            attn_out, attn_weights = self.attn(normed, normed, return_attention_scores=True)
        else:
            attn_out = self.attn(normed, normed)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return (x, attn_weights) if return_attention else x


class VIT(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.patch_encoder = PatchEncoder()
        self.encoders = [TransformerEncoder() for _ in range(NUM_LAYERS)]
        self.head = tf.keras.Sequential([
            layers.Flatten(),
            Dense(MLP_HEAD_UNITS, activation='relu'),
            Dense(NUM_CLASSES, activation='softmax')
        ])

    def call(self, x):
        x = self.patch_encoder(x)
        for encoder in self.encoders:
            x = encoder(x)
        return self.head(x)

    def get_attention_rollout(self, x):
        """Improved Attention Rollout with layer weighting for better tumor localization"""
        x = self.patch_encoder(x)
        attentions = []
        
        for encoder in self.encoders:
            x, attn = encoder(x, return_attention=True)
            # Average across heads: (batch, num_patches, num_patches)
            attn = tf.reduce_mean(attn, axis=1)
            attentions.append(attn)

        # Weighted attention rollout - later layers weighted more (more class-specific)
        n_layers = len(attentions)
        rollout = None
        
        for i, attn in enumerate(attentions):
            # Exponential layer weighting - later layers more important
            layer_weight = np.exp((i - n_layers + 1) / 3.0)
            
            # Smaller residual for better discrimination (0.2 instead of 0.5)
            attn_with_residual = 0.8 * attn + 0.2 * tf.eye(attn.shape[-1])
            
            if rollout is None:
                rollout = attn_with_residual * layer_weight
            else:
                rollout = tf.matmul(attn_with_residual, rollout) * layer_weight

        # Use mean attention per patch (better than sum for localization)
        mask = tf.reduce_mean(rollout, axis=1)
        
        grid_size = IMAGE_SIZE // PATCH_SIZE
        mask = tf.reshape(mask, (-1, grid_size, grid_size))
        
        # Robust normalization using percentiles to handle outliers
        mask_np = mask.numpy()
        p_low = np.percentile(mask_np, 2)
        p_high = np.percentile(mask_np, 98)
        mask_np = np.clip(mask_np, p_low, p_high)
        mask_np = (mask_np - p_low) / (p_high - p_low + 1e-8)
        
        return mask_np


# Helper functions for segmentation
def get_kmeans_mask(image, k=3):
    """K-Means clustering for edge segmentation"""
    img = np.stack([image.squeeze()]*3, axis=-1) if image.ndim < 3 or image.shape[-1] == 1 else image
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.reshape(image.shape[:2])


def extract_brain_mask(image):
    """Extract brain region, excluding skull and background"""
    img = image.squeeze()
    if img.max() > 1:
        img = img.astype(np.uint8)
    else:
        img = (img * 255).astype(np.uint8)
    
    # Threshold to get foreground (brain + skull)
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    
    # Flood fill from corners to remove external background
    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = binary.copy()
    
    # Flood fill from all four corners
    for seed in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
        cv2.floodFill(flood, mask.copy(), seed, 0)
    
    brain_region = flood
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_region = cv2.morphologyEx(brain_region, cv2.MORPH_CLOSE, kernel, iterations=2)
    brain_region = cv2.morphologyEx(brain_region, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Erode slightly to remove skull edge
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    brain_region = cv2.erode(brain_region, kernel_small, iterations=2)
    
    return brain_region > 0


def adaptive_attention_threshold(attention_map, brain_mask=None):
    """Compute adaptive threshold based on attention distribution within brain"""
    if brain_mask is not None:
        attn_in_brain = attention_map[brain_mask]
    else:
        attn_in_brain = attention_map.flatten()
    
    if len(attn_in_brain) == 0:
        return np.percentile(attention_map, 75)
    
    # Use Otsu's method on attention values
    attn_normalized = ((attn_in_brain - attn_in_brain.min()) / 
                       (attn_in_brain.max() - attn_in_brain.min() + 1e-8) * 255).astype(np.uint8)
    
    thresh_val, _ = cv2.threshold(attn_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to original scale
    threshold = attn_in_brain.min() + (thresh_val / 255.0) * (attn_in_brain.max() - attn_in_brain.min())
    
    # Ensure threshold is reasonable (between 50th and 85th percentile)
    p50 = np.percentile(attn_in_brain, 50)
    p85 = np.percentile(attn_in_brain, 85)
    threshold = np.clip(threshold, p50, p85)
    
    return threshold


def find_tumor_cluster(km_labels, attention_map, display_img, brain_mask, k=4):
    """Find the cluster most likely to be the tumor based on multiple criteria"""
    best_cluster = None
    best_score = -np.inf
    
    for c in range(k):
        cluster = km_labels == c
        cluster_in_brain = cluster & brain_mask if brain_mask is not None else cluster
        
        cluster_size = np.sum(cluster_in_brain)
        
        # Skip very small or very large clusters
        total_brain = np.sum(brain_mask) if brain_mask is not None else IMAGE_SIZE * IMAGE_SIZE
        if cluster_size < 100 or cluster_size > total_brain * 0.6:
            continue
        
        # Multiple scoring criteria for tumor detection:
        # 1. High attention overlap (most important)
        attn_score = np.mean(attention_map[cluster_in_brain]) if cluster_size > 0 else 0
        
        # 2. Intensity variance (tumors often have heterogeneous texture)
        intensity_var = np.std(display_img[cluster_in_brain]) if cluster_size > 0 else 0
        
        # 3. Compactness (tumors are typically compact regions)
        contours, _ = cv2.findContours(cluster_in_brain.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-8) if perimeter > 0 else 0
        else:
            compactness = 0
        
        # 4. Centrality (tumors often in interior, not at edges)
        y_coords, x_coords = np.where(cluster_in_brain)
        if len(x_coords) > 0:
            center_y, center_x = IMAGE_SIZE // 2, IMAGE_SIZE // 2
            dist_from_center = np.sqrt((np.mean(x_coords) - center_x)**2 + 
                                       (np.mean(y_coords) - center_y)**2)
            centrality = 1 - (dist_from_center / (IMAGE_SIZE * 0.5))
        else:
            centrality = 0
        
        # Combined score (attention is weighted highest)
        score = (attn_score * 3.0 + 
                 (intensity_var / 50) * 0.5 + 
                 compactness * 1.0 + 
                 centrality * 0.5)
        
        if score > best_score:
            best_score = score
            best_cluster = cluster_in_brain
    
    return best_cluster


def segment_tumor(model, img):
    img_batch = img[np.newaxis, ...]

    # Get prediction
    pred = model.predict(img_batch, verbose=0)
    pred_idx = np.argmax(pred)
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(np.max(pred) * 100)
    all_confidences = {CLASS_NAMES[i]: float(pred[0][i] * 100) for i in range(NUM_CLASSES)}

    # Normalize display image
    display = (img.squeeze() * 255 / (img.max() + 1e-8)).astype(np.uint8)

    # Get improved attention rollout
    attn = model.get_attention_rollout(img_batch)[0]
    attn = cv2.resize(attn, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

    # If NO TUMOR predicted, return empty segmentation
    if pred_label == 'notumor':
        empty_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        return {
            'prediction': pred_label,
            'confidence': confidence,
            'all_confidences': all_confidences,
            'attention_map': attn,
            'segmentation_mask': empty_mask,
            'display_image': display,
            'has_tumor': False
        }
    
    # Step 1: Extract brain mask (exclude skull and background)
    brain_mask = extract_brain_mask(display)
    
    # Step 2: Use adaptive threshold based on attention distribution within brain
    attn_threshold = adaptive_attention_threshold(attn, brain_mask)
    attn_mask = (attn > attn_threshold) & brain_mask
    
    # Step 3: K-Means clustering within brain region
    km_labels = get_kmeans_mask(display, k=4)
    
    # Step 4: Find best tumor cluster using multi-criteria scoring
    best_cluster = find_tumor_cluster(km_labels, attn, display, brain_mask, k=4)
    
    if best_cluster is None:
        # Fallback to attention-only segmentation
        best_cluster = attn_mask
    
    # Step 5: Combine attention mask with best cluster
    # Use intersection to be more precise
    combined = attn_mask & best_cluster
    
    # If intersection is too small, use union with more weight on attention
    if np.sum(combined) < 100:
        brain_attn = attn[brain_mask] if np.any(brain_mask) else attn.flatten()
        p60 = np.percentile(brain_attn, 60) if len(brain_attn) > 0 else 0.5
        combined = attn_mask | (best_cluster & (attn > p60))
    
    # Ensure we stay within brain mask
    combined = combined & brain_mask
    
    # Step 6: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_u8 = combined.astype(np.uint8) * 255
    combined_u8 = cv2.morphologyEx(combined_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_u8 = cv2.morphologyEx(combined_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 7: Keep only the largest connected component
    n, labels_map, stats, centroids = cv2.connectedComponentsWithStats(combined_u8)
    if n > 1:
        # Filter out very small components
        valid_components = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 50:  # Minimum area threshold
                # Score by area and attention overlap
                comp_mask = labels_map == i
                attn_in_comp = np.mean(attn[comp_mask])
                valid_components.append((i, area * attn_in_comp))
        
        if valid_components:
            # Select component with highest score
            best_comp = max(valid_components, key=lambda x: x[1])[0]
            final_mask = (labels_map == best_comp).astype(np.uint8) * 255
            tumor_centroid = centroids[best_comp]
        else:
            final_mask = (combined_u8 > 0).astype(np.uint8) * 255
            tumor_centroid = [IMAGE_SIZE // 2, IMAGE_SIZE // 2]
    else:
        final_mask = (combined_u8 > 0).astype(np.uint8) * 255
        tumor_centroid = [IMAGE_SIZE // 2, IMAGE_SIZE // 2]
    
    # Step 8: Final smoothing with convex hull approximation for cleaner edges
    if np.any(final_mask):
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get convex hull for smoother boundary
            hull_mask = np.zeros_like(final_mask, dtype=np.uint8)
            for contour in contours:
                hull = cv2.convexHull(contour)
                cv2.drawContours(hull_mask, [hull], 0, 255, -1)
            # Blend original and hull for balance between accuracy and smoothness
            final_mask = ((final_mask.astype(np.float32) + hull_mask.astype(np.float32)) / 2)
            final_mask = (final_mask > 100).astype(np.uint8) * 255

    tumor_area = np.sum(final_mask > 0)
    tumor_percentage = (tumor_area / (IMAGE_SIZE * IMAGE_SIZE)) * 100

    return {
        'prediction': pred_label,
        'confidence': confidence,
        'all_confidences': all_confidences,
        'attention_map': attn,
        'segmentation_mask': final_mask,
        'display_image': display,
        'has_tumor': True,
        'tumor_area': int(tumor_area),
        'tumor_percentage': float(tumor_percentage),
        'tumor_centroid': [float(tumor_centroid[0]), float(tumor_centroid[1])]
    }


def preprocess_image(image_bytes):
    """Preprocess uploaded image for model input"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=-1)
    return img


def create_3d_tumor_data(segmentation_mask, attention_map, num_slices=32):
    """Generate 3D tumor volume data from 2D segmentation"""
    h, w = segmentation_mask.shape
    volume = np.zeros((num_slices, h, w), dtype=np.float32)
    
    if np.sum(segmentation_mask) == 0:
        return volume.tolist()
    
    mask_normalized = segmentation_mask.astype(np.float32) / 255.0
    center_slice = num_slices // 2
    
    for z in range(num_slices):
        distance_from_center = abs(z - center_slice)
        scale = np.sqrt(max(0, 1 - (distance_from_center / center_slice) ** 2))
        
        if scale > 0.1:
            kernel_size = max(1, int((1 - scale) * 15))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            if kernel_size > 1:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                eroded = cv2.erode(mask_normalized, kernel, iterations=1)
            else:
                eroded = mask_normalized
            
            volume[z] = eroded * scale * (0.5 + 0.5 * attention_map)
    
    return volume.tolist()


def load_model_weights(model, weights_path):
    """Load weights into model - requires building the model first"""
    # Build the model with a dummy input
    dummy_input = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32)
    _ = model(dummy_input)
    # Now load the weights
    model.load_weights(weights_path)
    return model
