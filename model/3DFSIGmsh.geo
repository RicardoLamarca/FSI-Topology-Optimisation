SetFactory("OpenCASCADE");

lc = 0.005;

// Point Definitions
Point(1) = {1.000000, 0.001260, 0, lc};
Point(2) = {0.950000, 0.009990, 0, lc};
Point(3) = {0.900000, 0.018570, 0, lc};
Point(4) = {0.800000, 0.034360, 0, lc};
Point(5) = {0.700000, 0.048670, 0, lc};
Point(6) = {0.600000, 0.060860, 0, lc};
Point(7) = {0.500000, 0.070360, 0, lc};
Point(8) = {0.400000, 0.076670, 0, lc};
Point(9) = {0.300000, 0.078800, 0, lc};
Point(10) = {0.250000, 0.077840, 0, lc};
Point(11) = {0.200000, 0.075050, 0, lc};
Point(12) = {0.150000, 0.069880, 0, lc};
Point(13) = {0.100000, 0.061390, 0, lc};
Point(14) = {0.075000, 0.055420, 0, lc};
Point(15) = {0.050000, 0.047490, 0, lc};
Point(16) = {0.025000, 0.036480, 0, lc};
Point(17) = {0.012500, 0.028100, 0, lc};
Point(18) = {0.000000, 0.000000, 0, lc};
Point(19) = {0.012500, -0.023420, 0, lc};
Point(20) = {0.025000, -0.030600, 0, lc};
Point(21) = {0.050000, -0.039050, 0, lc};
Point(22) = {0.075000, -0.043740, 0, lc};
Point(23) = {0.100000, -0.046710, 0, lc};
Point(24) = {0.150000, -0.049600, 0, lc};
Point(25) = {0.200000, -0.050250, 0, lc};
Point(26) = {0.250000, -0.049420, 0, lc};
Point(27) = {0.300000, -0.047500, 0, lc};
Point(28) = {0.400000, -0.041670, 0, lc};
Point(29) = {0.500000, -0.034180, 0, lc};
Point(30) = {0.600000, -0.025880, 0, lc};
Point(31) = {0.700000, -0.017590, 0, lc};
Point(32) = {0.800000, -0.009920, 0, lc};
Point(33) = {0.900000, -0.003510, 0, lc};
Point(34) = {0.950000, -0.001010, 0, lc};
Point(35) = {1.000000, -0.001260, 0, lc};
Point(40) = {-0.5, -0.5, 0, 0.05};
Point(41) = {-0.5, 0.5, 0, 0.05};
Point(42) = {2, 0.5, 0, 0.05};
Point(43) = {2, -0.5, 0, 0.05};
Point(83) = {-0.5, -0.5, 2, 0.05};
Point(84) = {-0.5, 0.5, 2, 0.05};
Point(85) = {2, 0.5, 2, 0.05};
Point(86) = {2, -0.5, 2, 0.05};

BSpline(1) = {1:17, 18, 19:35, 1};

Line(27) = {41, 40};
Line(28) = {40, 83};
Line(29) = {83, 84};
Line(30) = {84, 41};
Line(31) = {83, 86};
Line(32) = {43, 86};
Line(33) = {86, 85};
Line(34) = {85, 42};
Line(35) = {42, 43};
Line(36) = {40, 43};
Line(37) = {84, 85};
Line(38) = {41, 42};

// Fluid Contour Surfaces
Curve Loop(3) = {28, 29, 30, 27}; Plane Surface(104) = {3};
Curve Loop(4) = {31, 33, -37, -29}; Plane Surface(105) = {4};
Curve Loop(5) = {36, 32, -31, -28}; Plane Surface(107) = {5};
Curve Loop(6) = {32, 33, 34, 35}; Plane Surface(106) = {6};
Curve Loop(7) = {36, -35, -38, 27}; Plane Surface(108) = {7};
Curve Loop(8) = {30, 38, -34, -37}; Plane Surface(109) = {8};

// ROTATE THE AIRFOIL CURVE
rotated_curve[] = Rotate { {0, 0, 1}, {0, 0, 0}, -10.0 * Pi / 180.0 } { Duplicata{ Curve{1}; } };

// Create Airfoil Surface FROM ROTATED CURVE
Curve Loop(2) = {rotated_curve[0]};
Plane Surface(2) = {2};

// Airfoil Extrusion
v[] = Extrude {0, 0, 1.5} {
  Surface{2};
  Layers{200};
};
// Define Volumes
// Surface Loop for outer box
Surface Loop(10) = {105, 104, 109, 108, 107, 106}; // Use new tags 104-109
Volume(2) = {10};
Surface{2,v[0],v[2]} In Volume {2};
// Coherence
Coherence;

// Define Wing Physical Groups
Physical Volume("Wing") = {v[1]};
Physical Surface("Wing") = {v[2]};
Physical Surface("fixed") = {2}; //
Physical Surface("interface") = {v[2],v[0]};

// Define Fluid Physical Groups
Physical Surface("inlet") = {116};
Physical Surface("outlet") = {114};
Physical Surface("NoSlip") = {115, 113, 112, 117};
Physical Volume("Fluid") = {2};

Mesh 3;
Save "3DFSImeshmid.msh";
