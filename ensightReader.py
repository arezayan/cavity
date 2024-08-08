# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:09:09 2024

@author: Amirreza
"""

import ensightreader
import numpy as np

case = ensightreader.read_case("E:/FOAM_PINN/cavHeat/2D_lamin_over_box/postProcessing/surfaces/zNormal/zNormal.case")
geofile = case.get_geometry_model()

part_names = geofile.get_part_names()           # ["internalMesh", ...]
part = geofile.get_part_by_name(part_names[0])
N = part.number_of_nodes

with geofile.open() as fp_geo:
    node_coordinates = part.read_nodes(fp_geo)  # np.ndarray((N, 3), dtype=np.float32)

variable = case.get_variable("UMean")

with variable.mmap_writable() as mm_var:
    data = variable.read_node_data(mm_var, part.part_id)
    data[:] = np.sqrt(data)                     # transform variable data in-place
