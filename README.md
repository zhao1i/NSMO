# USMO
Unsupervised Stereo Matching Optimization with Heterogeneous Smoothness for Satellite Image 3D Reconstruction


This project implements an unsupervised three-stage optimization framework for disparity map refinement, tailored to the characteristics of high-resolution satellite imagery and urban 3D reconstruction tasks. By combining region-based MRF modeling with heterogeneity-aware energy formulation and iterative edge-preserving filtering, this method improves the quality of disparity maps generated by traditional stereo matching algorithms.

---

## 🚀 Key Features

### 🧩 Step 1: Superpixel-Guided MRF Graph Construction

**Function:** `buildMRFGraph(D, IL, numSuperpixels)`

- Performs SLIC superpixel segmentation on the left image `IL`.
- Constructs an MRF graph where each superpixel is a node.
- Extracts mean intensity, disparity, spatial centroid, and local gradient per node.
- Builds a 4-neighbor adjacency matrix between superpixels.
- Generates an edge-aware mask (`edge_mask`) to guide subsequent adaptive smoothness.

### 🔧 Step 2: Region-Aware Energy Function

**Function:** `energyAndGradient(D, IL, IR, lambda_matrix)`

- Defines a heterogeneity-aware energy function with two terms:
  - **Data term:** Ensures stereo consistency between left and right images.
  - **Smoothness term:** Promotes regional consistency using adaptive weights.
- Computes total energy `E` and its gradient with respect to disparity.

### 🔁 Step 3: Gradient-Filter Iterative Optimization

**Function:** `lmm(Fk, JFk, x0, IL, IR, edge_mask)`

- Minimizes energy via gradient descent with line search.
- Applies bilateral filtering after each update to preserve edges and reduce noise.
- Converges to a refined disparity map using gradient norm, step decay, and max iterations.

---

## 📂 Project Structure

```
├── buildMRFGraph.m          % Superpixel segmentation and MRF graph construction
├── energyAndGradient.m      % Heterogeneity-aware energy and gradient computation
├── lmm.m                    % Iterative optimization using gradient + filtering
├── bilateral_filter.m       % Custom bilateral filter (optional)
├── sample_data/             % Sample stereo image and disparity inputs
└── README.md                % This file
```

---

## 📦 Dependencies

- MATLAB R2020b or newer
- Image Processing Toolbox
- (Optional) Your own stereo image pair and initial disparity map

---

## 🧪 Example Usage

```matlab
% Load stereo images and initial disparity
IL = imread('left.png');
IR = imread('right.png');
D0 = double(imread('init_disp.png'));

% Step 1: Build graph and edge mask
[graph_nodes, graph_edges, edge_mask, sp_labels] = buildMRFGraph(D0, IL, 200);

% Step 2 & 3: Optimize disparity
[x_opt, energy, num_iter] = lmm(@(x) energyAndGradient(x, IL, IR, lambda_matrix), ...
                                @(x) energyAndGradient(x, IL, IR, lambda_matrix), ...
                                D0, IL, IR, edge_mask);
```

---

## 📚 Citation

If you use this code in your research, please cite:

**Unsupervised Stereo Matching Optimization with Heterogeneous Smoothness for Satellite Image 3D Reconstruction**  
Bingqian Zhou, Li Zhao, et al.  
*IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025.*
