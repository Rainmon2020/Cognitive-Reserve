import os
import nibabel as nib
import numpy as np
import scipy.io

input_dir = '/v16data/user_data/lym/CA_Result/3d_gii'
output_dir = '/v16data/user_data/lym/CA_Result/rfmri_fs'


num_sub = 483
TR = 240

for i in range(241, num_sub+1):
    sub_id = f'CA{str(i).zfill(4)}'
    mat_id = f'CA{str(i-240).zfill(4)}'
    for hemi in ['lh', 'rh']:
        num_vertices = 0
        vertex_data = None

        for t in range(0, TR):
            filename = f'sub-{sub_id}_{hemi}_func_midthickness_{t}.gii'
            file_path = os.path.join(input_dir, filename)
            gii_img = nib.load(file_path)
            num_vertices = max(num_vertices, gii_img.darrays[0].data.size)

            if vertex_data is None:
                vertex_data = np.zeros((num_vertices, TR))

            current_vertex_data = np.zeros(num_vertices)
            current_vertex_data[:gii_img.darrays[0].data.size] = gii_img.darrays[0].data
            vertex_data[:num_vertices, t] = current_vertex_data

        output_filename = os.path.join(output_dir, f'sub-{mat_id}_rfMRI_timeseries-{hemi}.mat')
        scipy.io.savemat(output_filename, {'timeseries': vertex_data})
        
        
