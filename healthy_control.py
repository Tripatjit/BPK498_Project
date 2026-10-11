from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

#Retreive Atlas and data

dataset = datasets.fetch_atlas_yeo_2011()
atlas_filename = dataset.maps
labels = dataset.labels
look_up_table = dataset.lut

print(f"Atlas ROIs are located in nifti image (4D) at: {atlas_filename}")

data = datasets.fetch_development_fmri(n_subjects=1, reduce_confounds=True)
fmri_filenames = data.func[0]
reduced_confounds = data.confounds[0] 

#Extract signals

masker = NiftiLabelsMasker(
    labels_img=atlas_filename,
    lut=look_up_table,
    standardize_confounds=True,
    standardize="zscore_sample",
    memory="nilearn_cache",
    verbose=1,
)

time_series = masker.fit_transform(fmri_filenames, confounds=reduced_confounds)

print(time_series)

time_series_nft = masker.inverse_transform(time_series)
time_series_nft.to_filename("time_series_nft.nii.gz")


