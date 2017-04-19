#!/usr/bin/env python
from .utils import (get_dsi_studio_ODF_geometry, 
    pairwise_distances, angle_between, get_area_3d)
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# ONLY WORKS ASSUMING RAS+, ONLY USED IN write_solutions
neighbor_targets = {
    # z = 0
    "la":  ((-1, 1, 0),( 0, 2, 1)),
    "a":   (( 0, 1, 0),( 1, 2, 1)),
    "ra":  (( 1, 1, 0),( 2, 2, 1)),
    
    "r":   (( 1, 0, 0),( 2, 1, 1)),
    "l":   ((-1, 0, 0),( 0, 1, 1)),
    
    "lp":  ((-1,-1, 0),( 0, 0, 1)),
    "p":   (( 0,-1, 0),( 1, 0, 1)),
    "rp":  (( 1,-1, 0),( 2, 0, 1)),
    
    # z = 1
    "las": ((-1, 1, 1),( 0, 2, 2)),
    "as":  (( 0, 1, 1),( 1, 2, 2)),
    "ras": (( 1, 1, 1),( 2, 2, 2)),
    
    "ls":  ((-1, 0, 1),( 0, 1, 2)),
    "s":   (( 0, 0, 1),( 1, 1, 2)),
    "rs":  (( 1, 0, 1),( 2, 1, 2)),
    
    "rps": (( 1,-1, 1),( 2, 0, 2)),
    "ps":  (( 0,-1, 1),( 1, 0, 2)),
    "lps": ((-1,-1, 1),( 0, 0, 2)),
    
    # z = -1
    "rai": (( 1, 1,-1),( 2, 2, 0)),
    "ai":  (( 0, 1,-1),( 1, 2, 0)),
    "lai": ((-1, 1,-1),( 0, 2, 0)),
    
    "ri":  (( 1, 0,-1),( 2, 1, 0)),
    "i":   (( 0, 0,-1),( 1, 1, 0)),
    "li":  ((-1, 0,-1),( 0, 1, 0)),

    "rpi": (( 1,-1,-1),( 2, 0, 0)),
    "pi":  (( 0,-1,-1),( 1, 0, 0)),
    "lpi": ((-1,-1,-1),( 0, 0, 0))
}

singleODF_header= """

subroutine %s_prob(X, G, P)
       implicit none
       real(8), intent(in), dimension(0:%d) :: X 
       real(8), intent(in), dimension(0:%d,0:%d) :: G 
       real(8), intent(out) :: P
       P=0.

""" 

doubleODF_header= """

subroutine %s_prob(X, Y, G, P)
       implicit none
       real(8), intent(in), dimension(0:%d) :: X
       real(8), intent(in), dimension(0:%d) :: Y
       real(8), intent(in), dimension(0:%d,0:%d) :: G
       real(8), intent(out) :: P

       P=0.

"""

def get_expressions(odf_order, angle_max, step_size, nbr_name, odf_vertex_num):

    odf_vertices, odf_faces = get_dsi_studio_ODF_geometry(odf_order)
    compatible_angles = pairwise_distances(odf_vertices,metric=angle_between) < angle_max
    compatible_indices = np.array([np.flatnonzero(r) for r in compatible_angles])
    
    def compute_area_by_target(target, path, transition_matrix, 
                               compatible = range(len(odf_vertices))):
        for j in compatible:
            direction = odf_vertices[j]
            vol, target_start, target_end = get_area_3d((0,0,0), (1,1,1), target[0], 
                                                     target[1], direction, step_size)
            transition_matrix[(j,)+path] = vol
            if vol > 0:
                transition_matrix = compute_area_by_target([target_start, target_end], 
                                (j,)+path, transition_matrix, compatible=compatible_indices[j])
        return transition_matrix

    transition_matrix = defaultdict(float)
    
    # The first iteration is done outside the recursive function
    direction = odf_vertices[odf_vertex_num]
    nbr_target = neighbor_targets[nbr_name]
    vol, target_start, target_end = get_area_3d((0,0,0), (1,1,1), nbr_target[0], 
                                             nbr_target[1], direction, step_size)

    # if there is any volume, keep going
    if vol > 0:
        transition_matrix[(nbr_name, odf_vertex_num)] = vol
        transition_matrix = compute_area_by_target([target_start, target_end], 
                        (nbr_name, odf_vertex_num), transition_matrix,
                        compatible=compatible_indices[odf_vertex_num])
    return transition_matrix

def write_files(odf_order, angle_max, step_size):

    odf_vertices, odf_faces = get_dsi_studio_ODF_geometry(odf_order)
    all_verts = odf_vertices.shape[0]
    unique_verts = all_verts // 2 # only half of the odf_vertices contain unique diffusion data
    N = unique_verts - 1

    # Convert turning angle sequences into fortran code
    def singleODF_to_fortran(path, weight):
        if weight == 0: return ""
        #Determine the probability of taking that particular sequence of angles 
        first_step = path[-1] if len(path) == 2 else path[0]
        expr = ["%.25f"%weight, "*", "X(%d)" % ( first_step%unique_verts)]
        if len(path) == 2: return "      P=P+" + " ".join(expr)
        expr = expr + ["*",  "G(%d,%d)" % (path[-3]%unique_verts, path[-1]%unique_verts)]

        for k in range(len(path)-3): # conditional probabilities for the rest of the steps
            expr = expr + ["*", "G(%d,%d)" % (path[k]%unique_verts, path[k+1]%unique_verts) ]
        
        # fit the fortran line limit
        lines = ["      P=P+" ]
        for token in expr:
            line_len = len(lines[-1])
            token_len = len(token)
            if line_len + token_len > 70:
                lines[-1] += " &"
                lines.append("        ")
            lines[-1] += token

        return "\n".join(lines) 

    def doubleODF_to_fortran(path, weight):
        if weight == 0: return ""

        #Determine the probability of taking that particular sequence of angles 
        first_step = path[-1] if len(path) == 2 else path[0]
        expr = ["%.25f"%weight, "*", "X(%d)" % ( 
            first_step%unique_verts),"*", "Y(%d)" % ( path[-1]%unique_verts)]
        if len(path) == 2: return "      P=P+" + " ".join(expr)
        expr = expr + ["*",  "G(%d,%d)" % (path[-3]%unique_verts, path[-1]%unique_verts)]

        for k in range(len(path)-3): # conditional probabilities for the rest of the steps
            expr = expr + ["*", "G(%d,%d)" % (path[k]%unique_verts, path[k+1]%unique_verts) ]
        
        # fit the fortran line limit
        lines = ["      P=P+" ]
        for token in expr:
            line_len = len(lines[-1])
            token_len = len(token)
            if line_len + token_len > 70:
                lines[-1] += " &"
                lines.append("        ")
            lines[-1] += token

        return "\n".join(lines) 


    txt_step_size = "%.02f"%step_size
    txt_step_size = txt_step_size.replace(".","_") # No periods in module names
    suffix = "_%s_ss%s_am%d.f90" % (odf_order,txt_step_size,angle_max)

    doubleODF_file = open("doubleODF" + suffix, "w")
    singleODF_file = open("singleODF" + suffix, "w")

    for target_code in tqdm(list(neighbor_targets.keys())):
        print("Solving for ", target_code)
        # Write the function definition
        doubleODF_file.write(doubleODF_header % (target_code,N,N,N,N))
        singleODF_file.write(singleODF_header % (target_code,N,N,N))
        
        all_expressions = {}
        for starting_direction in tqdm(range(all_verts)):
            all_expressions.update( get_expressions(odf_order, angle_max, step_size,
                target_code, starting_direction))

        singleODF_exprs = []
        doubleODF_exprs = []
        for k,v in all_expressions.items():
            if v == 0: continue
            singleODF_exprs.append(singleODF_to_fortran(k,v))
            doubleODF_exprs.append(doubleODF_to_fortran(k,v))

        # Write out the expressions to files
        for na_expr in singleODF_exprs:
            singleODF_file.write(na_expr+"\n")
        singleODF_file.write("end subroutine\n\n\n\n")
        for expr in doubleODF_exprs:
            doubleODF_file.write(expr+"\n")
        doubleODF_file.write("end subroutine\n\n\n\n")
    doubleODF_file.close()
    singleODF_file.close()


