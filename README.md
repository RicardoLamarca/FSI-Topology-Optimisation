# 🌊 Navier-Stokes Immersed Body Topology Optimization

![Julia](https://img.shields.io/badge/Julia-1.7%2B-9558B2?style=for-the-badge&logo=julia)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

A self-contained **Julia** script for performing **2D & 3D Topology Optimization** on structures immersed in Navier-Stokes fluid flow.

This project utilizes the **Method of Moving Asymptotes (MMA)** to optimize the material distribution of a body, minimizing compliance (maximizing stiffness) under dynamic fluid-structure interaction loads.

---

## ✨ Key Features

* **📦 Self-Contained Architecture:** All solver logic, Finite Element state maps, and optimization functionals are packed into a single file. No need to manage complex local packages.
* **🪟 Zero-Config Windows Support:** Automatically detects Windows environments and pins `P4est_jll` to version 2.8.1 to prevent binary incompatibility crashes.
* **⚡ Automated Dependency Setup:** First-run auto-installation of required packages (`Gridap`, `NLopt`, `WriteVTK`, etc.).
* **💧 Dynamic Fluid Loads:** Computes Navier-Stokes fluid loads dynamically on the solid interface during the optimization loop.

---

## 🛠️ Prerequisites

Before running the simulation, ensure you have the following installed:

1. **[Julia](https://julialang.org/downloads/)** (v1.7 or higher).
2. **[Gmsh](https://gmsh.info/)**: Required to generate the `.msh` mesh files.
3. **[ParaView](https://www.paraview.org/)**: To visualize the resulting `.vtu` files.

---

## 🕸️ Mesh Preparation (Crucial Step)

The most important step is creating a valid `.msh` file using Gmsh. The Julia script relies on **specific Physical Group names (tags)** to apply boundary conditions correctly.

### 🏷️ Mandatory Labels (Physical Groups)
Your `.geo` file **must** include the following Physical Curves/Surfaces/Volumes with these exact names:

| Tag Name | Type (2D) | Type (3D) | Description |
| :--- | :--- | :--- | :--- |
| `inlet` | Curve | Surface | Where the fluid enters the domain. |
| `outlet` | Curve | Surface | Where the fluid exits. |
| `noSlip` | Curve | Surface | Walls where fluid velocity is zero. |
| `fluid` | Surface | Volume | The fluid domain. |
| `solid` | Surface | - | The design domain (structure). |
| `interface` | Curve | Surface | Boundary between fluid and solid. |
| `fixed` | Point/Curve | Surface | Dirichlet BC for the structure. |
## 🚀 Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Generate your Mesh:**
    Create your `.msh` file using Gmsh (as explained above) and place it in the project directory.

3.  **Configure the Script:**
    Open `2DFSITOPOPT.jl` and modify the path to your mesh file at the bottom of the script (approx. line 930):
    ```julia
    # Line 930
    mesh_file_path = "./your_generated_mesh.msh" 
    ```

4.  **Run the Optimization:**
    ```bash
    julia 2DFSITOPOPT.jl
    ```
    > **Note:** The first run may take **5-10 minutes** as Julia compiles the physics engine and installs dependencies. Subsequent runs will be much faster.

---

## 📊 Visualizing Results

The simulation outputs `.vtu` files to the `./results` folder. Open these in **ParaView** to analyze:

* **$\rho_h$ (Density):** The material distribution ($1.0 = \text{solid}$, $0.0 = \text{void}$).
* **$u_h$ (Displacement):** The structural deformation.
* **$p_h$ (Pressure):** The fluid pressure field interacting with the structure.

---

### 🤝 Contribution

Contributions are welcome! Please feel free to submit a Pull Request.
