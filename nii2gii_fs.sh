for i in {478..483}; do
    sid=$(printf "sub-CA%04d" $i)
    echo "Processing ${sid}"

    for hemisphere in 'lh' 'rh'; do
        echo "Processing ${hemisphere}"

        for t in {0..239}; do
            t=$(printf $t)
            echo "Processing ${t}"
            mri_vol2surf --src "/v16data/user_data/lym/CA_Result/3d_nii/${sid}_task-rest_T1W_bold_3d_${t}.nii" \
                         --out "/v16data/user_data/lym/CA_Result/3d_gii/${sid}_${hemisphere}_func_midthickness_${t}.gii" \
                         --regheader "${sid}" \
                         --hemi "${hemisphere}" \
                         --trgsubject fsaverage
        done
    done
done
