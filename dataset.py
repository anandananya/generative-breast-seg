import glob
import logging
import os
import socket
import urllib.request
from builtins import isinstance, str

import pandas as pd
import tabulate
import torch
from monai.data import Dataset

from data_utils import get_label_encodings
from tcia_download import download_collection

logger = logging.getLogger()
series_map = {
    "ax t1": "t1",  # non fat saturated t1 https://www.reddit.com/r/DukeDCEMRIData/comments/zf9ati/what_do_the_dicom_stacks_represent_specifically/"
    "ax dyn pre": "pre",  # fat saturated t1
    "ax dyn 1st pass": "post_1",
    "ax dyn 2nd pass": "post_2",
    "ax dyn 3rd pass": "post_3",
    "Ph2/ax 3d dyn": "post_2",
    "ax 3d dyn": "pre",
    "Ph1/ax 3d dyn": "post_1",
    "Ph3/ax 3d dyn": "post_3",
    "Ph1/ax 3d dyn MP": "post_1",
    "Ph2/ax 3d dyn MP": "post_2",
    "Ph3/ax 3d dyn MP": "post_3",
    "Ph4/ax 3d dyn MP": "post_4",
    "ax 3d dyn MP": "pre",
    "ax dynamic": "pre",
    "Ph2/ax dynamic": "post_2",
    "Ph3/ax dynamic": "post_3",
    "Ph1/ax dynamic": "post_1",
    "ax dyn 4th pass": "post_4",
    "Ax Vibrant MultiPhase": "pre",
    "Ax 3D T1 NON FS": "t1",
    "Ph3/Ax Vibrant MultiPhase": "post_3",
    "Ph4/Ax Vibrant MultiPhase": "post_4",
    "Ph2/Ax Vibrant MultiPhase": "post_2",
    "Ph1/Ax Vibrant MultiPhase": "post_1",
    "ax 3d t1 bilateral": "t1",
    "ax dyn": "pre",
    "Ph1/ax dyn": "post_1",
    "Ph4/ax 3d dyn": "post_4",
    "Ph2/ax dyn": "post_2",
    "Ph3/ax dyn": "post_3",
    "Ph4/ax dyn": "post_4",
    "ax 3d dyn 2nd pass": "post_2",
    "ax 3d dyn pre": "pre",
    "ax 3d dyn 3rd pass": "post_3",
    "ax 3d dyn 4th pass": "post_4",
    "ax 3d dyn 1st pass": "post_1",
    "ax t1 +c": "t1",
    "ax t1 tse": "t1",  # tse: turbo spin echo
    "ax t1 tse +c": "t1",
    "ax t1 3mm": "t1",
    "1st ax dyn": "post_1",
    "2nd ax dyn": "post_2",
    "3rd ax dyn": "post_3",
    "AX IDEAL Breast": "t1",  # iterative decomposition of water and fat with echo asymmetry and least-squares estimation (IDEAL) https://www.ajronline.org/doi/pdf/10.2214/AJR.07.3182
    "ax 3d pre": "pre",
    "4th ax dyn": "post_4",
    "ax t1 pre": "t1",
    "Ph4/ax dynamic": "post_4",
    "t1_fl3d_tra_dynVIEWS_spair_2nd pass": "post_2",
    "t1_fl3d_tra_dynVIEWS_spair_4th pass": "post_4",
    "t1_fl3d_tra_dynVIEWS_spair_ pre": "pre",
    "t1_fl3d_tra_dynVIEWS_spair_3rd pass": "post_3",
    "t1_fl3d_tra_dynVIEWS_spair 1st pass": "post_1",
    "ax t1 2mm": "t1",
    "ax t1 repeat": "t1",
    "ax dyn 1st pas": "post_1",
}


def get_data_path():
    hostname = socket.gethostname()
    if hostname in [
        "cri16cn407.cri.uchicago.edu",
        "cri16cn406.cri.uchicago.edu",
    ]:
        return "/scratch/annawoodard/duke/preprocessed"
    return "/net/projects/cdac/annawoodard/duke/preprocessed"


def log_summary(tag, df, target):
    """
    Generates a summary log of the dataset with specific focus on series, studies and unique women. The log will be output in a tabulated format and is helpful for understanding dataset characteristics.

    Arguments:
        tag (str): A tag that represents the dataset.
        df (pandas.DataFrame): The dataset to be summarized. The DataFrame should have 'study_id', 'patient_id', 'series_id', and optionally, 'encoded_target' and 'target' columns.
        target (str, optional): The target column to focus on for generating the summary. If None, the function will summarize the dataset based on 'study_id', 'patient_id' and 'series_id'. If provided, it should be a column in the DataFrame. The summary will include counts per unique value in this column.

    The function will add a column 'encoded_target' to the dataframe if it is not already present and 'target' is specified. It will log a summary table containing counts of total series, studies and unique women, broken down by the unique values in the 'encoded_target' column.

    Note: This function logs the summary, it doesn't return anything.
    """
    if target is None:
        table = [
            len(df),
            df.study_id.nunique(),
            df.patient_id.nunique(),
        ]
        headers = [
            "series",
            "studies",
            "unique women",
        ]
        values = []
    else:
        if "encoded_target" not in df.columns:
            df["encoded_target"] = None
        values = sorted(df["encoded_target"].unique())
        table = []
        headers = []
        for value in values:
            table += [
                len(df[df["encoded_target"] == value]),
                df[df["encoded_target"] == value].series_id.nunique(),
                df[df["encoded_target"] == value].patient_id.nunique(),
            ]
            headers += [
                "series",
                "studies",
                "unique women",
            ]
    table_text = tabulate.tabulate([table], headers)
    table_width = len(table_text.split("\n")[0])
    tag = tag + " dataset summary"
    label_width = len(target) if target is not None else 10
    padding = table_width - label_width - 1
    top_row = [f"\n{tag} {'*' * padding}\n"]
    middle_row = [" " * len(tag) + " " * padding + "\n"]
    bottom_row = []
    for value in values:
        matching_targets = "|".join(
            [str(x) for x in df[df["encoded_target"] == value]["target"].unique()]
        )
        top_row += [
            " " * (35 - len(f"{target}={matching_targets}"))
            + f"{target}={matching_targets}"
        ]
        middle_row += [" " * (35 - len(f"encoded={value}")) + f"encoded={value}"]
        bottom_row += ["_" * 35 + " " * 2]
    logger.info(
        f"\n{tag} {'*' * padding}\n"
        + "".join(top_row)
        + "\n"
        + "".join(middle_row)
        + "\n"
        + "".join(bottom_row)
        + "\n"
        + table_text
        + "\n"
    )


def get_metadata(
    data_path,
    target,
    label_encoding=None,
    exclude=None,
    prescale=1.0,
    include_series=None,
    require_series=None,
):
    """
    Retrieve and filter metadata for a particular dataset based on various conditions.

    Arguments:
        data_path (str): The directory where the metadata.csv file is located.
        target (str): The target column name in the metadata that the analysis will focus on.
        label_encoding (dict, optional): A dictionary with original labels as keys and encoded labels as values. If None, it defaults to index-based encoding of unique labels.
        exclude (list, optional): A list of patient IDs to be excluded from the returned metadata.
        prescale (float, optional): Fraction of metadata to sample. If not 1.0, a random subset of the metadata will be sampled. Default is 1.0 which means no sampling.
        include_series (str or list, optional): Series descriptions to include in the returned metadata. If None, all series are included.
        require_series (str or list, optional): Series descriptions that must exist for a study to be included in the metadata. If None, all studies are included.

    Returns:
        pandas.DataFrame: The metadata DataFrame filtered based on the given conditions.

    Raises:
        FileNotFoundError: If the metadata.csv file does not exist at the given location.

    Note:
        This function filters the data using various hard-coded and input values, including:
        - Filtering out specific series based on the SeriesInstanceUID.
        - Removing data for specific patients based on their PatientID.
        - Adjusting labels for certain series using CleanedSeriesDescription.
        - Ensuring that the nifti files exist for each row in the metadata.
        - Dropping rows where the target column has a null value.
        - Encoding the target labels and adding a new 'encoded_target' column to the metadata.
    """
    if isinstance(include_series, str):
        include_series = [include_series]
    try:
        metadata = pd.read_csv(
            os.path.join(
                data_path,
                "_".join(include_series) if include_series is not None else "",
                "metadata.csv",
            )
        )
    # TODO: add a series description directory layer when downloading dicoms to match the postprocessed structure
    except FileNotFoundError:
        metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))
    # FIXME hack there are duplicates causing issues
    filter_series = [
        "1.3.6.1.4.1.14519.5.2.1.266216227826305484822453682925025426138",
        "1.3.6.1.4.1.14519.5.2.1.37337419554057095415343764757927450341",
        "1.3.6.1.4.1.14519.5.2.1.224916637743088409523159007986708320181",  # Breast_MRI_596, appears to be dupe image
        "1.3.6.1.4.1.14519.5.2.1.299122154354227147513399031558444741694",  # Breast_MRI_596, appears to be dupe image
        "1.3.6.1.4.1.14519.5.2.1.50433126667963956508264790315941584784",  # Breast_MRI_596, appears to be dupe image
        "1.3.6.1.4.1.14519.5.2.1.301984269451094267320716240972599525201",  # Breast_MRI_596, appears to be dupe image
    ]
    metadata = metadata[
        ~metadata.SeriesInstanceUID.str.contains("|".join(filter_series))
    ]
    # TODO FIXME HACK!!!!!
    # There is something wrong about these images-- dicom2nifti originally failed on them
    # Opening with the monai dicom reader succeeded but then the segmentation failed
    # filter_patients = [
    #     "Breast_MRI_056",
    #     "Breast_MRI_085",
    #     "Breast_MRI_042",
    #     "Breast_MRI_156",
    #     "Breast_MRI_864",
    # ]
    # metadata = metadata[~metadata.PatientID.str.contains("|".join(filter_patients))]
    # appear to be mislabeled as 'pre'
    for series_number, series_description in [
        (8.0, "post_1"),
        (11.0, "post_2"),
        (13.0, "post_3"),
        (15.0, "post_4"),
    ]:
        metadata.loc[
            (metadata.PatientID == "Breast_MRI_120")
            & (metadata.SeriesNumber == series_number),
            "CleanedSeriesDescription",
        ] = series_description
    logger.info(f"loaded {len(metadata)} examples")
    metadata = metadata[metadata.nifti_exists == True]

    if exclude is not None:
        original = metadata.nifti_path.nunique()
        metadata = metadata[~metadata["PatientID"].isin(exclude)]
        logger.info(
            f"dropped {original - metadata.nifti_path.nunique()} views from patients in the finetuning testing set"
        )
    if prescale:
        metadata = metadata.sample(frac=prescale)
    if include_series is not None:
        if isinstance(include_series, str):
            include_series = [include_series]
        metadata = metadata[~pd.isnull(metadata.CleanedSeriesDescription)]
        metadata = metadata[
            metadata.CleanedSeriesDescription.str.contains("|".join(include_series))
        ]
        logger.info(
            f"there are {len(metadata)} examples including series {'|'.join(include_series)}"
        )
    if require_series is not None:
        if isinstance(require_series, str):
            require_series = [require_series]
        require_series = set(require_series)
        passing_studies = []
        for study in metadata.StudyInstanceUID.unique():
            series = set(
                metadata[
                    metadata.StudyInstanceUID == study
                ].CleanedSeriesDescription.to_list()
            )
            if len(require_series & series) >= len(require_series):
                passing_studies.append(study)
        metadata = metadata[
            metadata.StudyInstanceUID.str.contains("|".join(passing_studies))
        ]
        logger.info(
            f"there are {len(metadata)} examples requiring series {'|'.join(require_series)}"
        )

    if target is not None:
        metadata = metadata[~pd.isnull(metadata[target])]
        logger.info(f"there are {len(metadata)} examples with non-null target {target}")
        # TODO implement regression as well as categorization
        metadata["target"] = metadata[target]
        for key, value in get_label_encodings(
            metadata[target], label_encoding, 1
        ).items():
            metadata.loc[
                metadata[target] == key,
                "encoded_target",
            ] = value

    metadata["study_id"] = metadata.StudyInstanceUID
    metadata["patient_id"] = metadata.PatientID

    return metadata


def drop_keys_from_dicts(dicts, keys):
    for d in dicts:
        for k in keys:
            d.pop(k, None)
    return dicts


class DukeSeriesDataset(Dataset):
    """Handles the Duke Breast Cancer MRI dataset. It subclasses the `Dataset` class in Monai.

    Attributes:
        data_path (str): The directory where the metadata.csv file is located.
        target (str): The target column name in the metadata that the analysis will focus on.
        transform (torch.nn.Identity): An Identity transform.
        metadata (pd.DataFrame): The metadata DataFrame.
        boxes (dict): Dictionary mapping each patient to a bounding box.
        tag (str): An identifying tag for logging purposes.
        label_encodings (dict): Dictionary for encoding the target labels.
        data (list): List of dictionaries. Each dictionary corresponds to a study and contains information such as path, patient_id, study_id, and target.

    Methods:
        num_classes(): Returns the number of unique classes in the label encodings.
        download(): Downloads and processes the data if it does not already exist.
    """

    def __init__(
        self,
        target=None,
        exclude: pd.core.series.Series = None,
        # image_size: int = (2016, 3808),
        prescale: float = 1.0,
        data_path: str = None,
        metadata=None,
        include_series=None,
        require_series=None,
        tag="",
        label_encodings=None,
        regression=False,
        download=False,
    ):
        """Initializes the `DukeSeriesDataset` with necessary parameters.

        Args:
            target (str, optional): The target column name in the metadata that the analysis will predict.
            exclude (pd.Series, optional): A series of patient IDs to be excluded from the dataset.
            prescale (float, optional): Fraction of metadata to sample. If not 1.0, a random subset of the metadata will be sampled. Default is 1.0 which means no sampling.
            data_path (str, optional): The directory where the metadata.csv file is located. Default is None which means a data path is automatically determined.
            metadata (pd.DataFrame, optional): A pandas DataFrame providing the metadata. If None, metadata will be loaded from the specified data_path.
            include_series (str or list, optional): Series descriptions to include in the returned metadata. If None, all series are included.
            require_series (str or list, optional): Series descriptions that must exist for a study to be included in the metadata. If None, all studies are included.
            tag (str, optional): An identifying tag for logging purposes.
            label_encodings (dict, optional): A dictionary with original labels as keys and encoded labels as values. If None, it defaults to index-based encoding of unique labels.
            regression (bool, optional): If True, prepares the dataset for regression tasks. Default is False.
            download (bool, optional): If True, downloads the data if it does not exist locally. Default is False.
        """
        self.target = target
        if data_path is None:
            data_path = get_data_path()
        self.data_path = data_path
        if download:
            self.download()
        self.transform = torch.nn.Identity()
        if metadata is None:
            self.metadata = get_metadata(
                data_path,
                target,
                label_encoding=label_encodings,
                exclude=exclude,
                prescale=prescale,
                include_series=include_series,
                require_series=require_series,
            )
        else:
            self.metadata = metadata
        self.tag = tag

        studies = []
        if not regression:
            if target is not None:
                label_encodings = get_label_encodings(
                    self.metadata[target], label_encodings
                )
        for _, row in self.metadata.iterrows():
            path = os.path.join(data_path, row.path)
            data = {
                "path": path,
                "image": path,
                "patient_id": row.PatientID,
                "series_id": row.SeriesInstanceUID,
                "study_id": row.StudyInstanceUID,
                "series_description": row.CleanedSeriesDescription,
                "box": self.boxes[row.PatientID],
            }
            if target is not None:
                data["target"] = row.target
                if regression:
                    data["encoded_target"] = row.target
                else:
                    data["encoded_target"] = int(row.encoded_target)
                data["image"] = path
            studies.append(data)
            print("appending " + path)
        log_summary(self.tag, pd.DataFrame(studies), target)
        drop_keys_from_dicts(studies, ["target"])
        self.label_encodings = label_encodings
        self.data = sorted(studies, key=lambda x: x["patient_id"])

    def num_classes(self):
        return len(set(self.label_encoding.values()))

    def download(self):
        if not os.path.isfile(
            os.path.join(self.data_path, "dicom", "per_series_metadata.csv")
        ):
            download_collection(
                "Duke-Breast-Cancer-MRI",
                "MR",
                os.path.join(self.data_path, "dicom"),
                cpus=10,
            )
        if not os.path.isfile(os.path.join(self.data_path, "filename_map.xlsx")):
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70226903/Breast-Cancer-MRI-filepath_filename-mapping.xlsx?api=v2",
                os.path.join(self.data_path, "filename_map.xlsx"),
            )
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70226903/Clinical_and_Other_Features.xlsx?api=v2",
                os.path.join(self.data_path, "clinical_and_other_features.xlsx"),
            )
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70226903/Annotation_Boxes.xlsx?api=v2",
                os.path.join(self.data_path, "annotation_boxes.xlsx"),
            )
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70226903/Imaging_Features.xlsx?api=v2",
                os.path.join(self.data_path, "imaging_features.xlsx"),
            )
            urllib.request.urlretrieve(
                "https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/287?passcode=36651c13b854907593b387dc22dec510833aa88c",
                os.path.join(self.data_path, "2d_breast_and_fgt_segmentations.nrrd"),
            )
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70226903/README.txt?version=2&modificationDate=1661571115801&api=v2",
                os.path.join(self.data_path, "readme.txt"),
            )
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70226903/segmentation_filepath_mapping.csv?version=1&modificationDate=1655149774283&api=v2",
                os.path.join(
                    self.data_path, "2d_breast_and_fgt_segmentations_filepath_map.csv"
                ),
            )
            urllib.request.urlretrieve(
                "https://wiki.cancerimagingarchive.net/download/attachments/70226903/Breast_Radiologist_Density_Assessments.xlsx?version=1&modificationDate=1655149776553&api=v2",
                os.path.join(
                    self.data_path, "breast_radiologist_density_assessments.xlsx"
                ),
            )
            urllib.request.urlretrieve(
                "https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/322?passcode=074f181da41cb15d0e64707e28738bf7b1b99a90",
                os.path.join(self.data_path, "3d_breast_and_fgt_segmentations.nrrd"),
            )
        if not os.path.isfile(os.path.join(self.data_path, "metadata.csv")):
            series_metadata = pd.read_csv(
                os.path.join(self.data_path, "dicom", "per_series_metadata.csv")
            )
            for original, cleaned in series_map.items():
                series_metadata.loc[
                    series_metadata.SeriesDescription == original,
                    "CleanedSeriesDescription",
                ] = cleaned

            series_metadata.to_csv(
                os.path.join(self.data_path, "dicom", "per_series_metadata.csv"),
                index=False,
            )

            annotations = pd.read_excel(
                os.path.join(self.data_path, "clinical_and_other_features.xlsx"),
                header=1,
            )
            annotations = annotations.drop(labels=0, axis=0)
            annotations.rename({"Patient ID": "PatientID"}, axis=1, inplace=True)

            # metadata now has one row per series; each row has per-patient annotations
            metadata = annotations.merge(
                series_metadata.set_index("PatientID"), on="PatientID"
            )

            metadata.to_csv(os.path.join(self.data_path, "metadata.csv"), index=False)


class DukeSegmentationDataset(DukeSeriesDataset):
    """Handles the Duke Breast Cancer MRI dataset for segmentation tasks.
       It subclasses the `DukeSeriesDataset` class.

    Attributes:
        data_path (str): The directory where the metadata.csv file is located.
        transform (torch.nn.Identity): An Identity transform.
        metadata (pd.DataFrame): The metadata DataFrame.
        tag (str): An identifying tag for logging purposes.
        label_encoding (dict): Dictionary for encoding the target labels.
        data (list): List of dictionaries. Each dictionary corresponds to a study and contains information such as path, patient_id, study_id, and target.
    """

    def __init__(
        self,
        exclude: pd.core.series.Series = None,
        # image_size: int = (2016, 3808),
        prescale: float = 1.0,
        data_path: str = None,
        metadata=None,
        include_series=None,
        require_series=None,
        tag="",
        label_encoding=None,
        download=False,
    ):
        """Initializes the `DukeSegmentationDataset` with necessary parameters.

        Args:
            exclude (pd.Series, optional): A series of patient IDs to be excluded from the dataset.
            prescale (float, optional): Fraction of metadata to sample. If not 1.0, a random subset of the metadata will be sampled. Default is 1.0 which means no sampling.
            data_path (str, optional): The directory where the metadata.csv file is located. Default is None which means a data path is automatically determined.
            metadata (pd.DataFrame, optional): A pandas DataFrame providing the metadata. If None, metadata will be loaded from the specified data_path.
            include_series (str or list, optional): Series descriptions to include in the returned metadata. If None, all series are included.
            require_series (str or list, optional): Series descriptions that must exist for a study to be included in the metadata. If None, all studies are included.
            tag (str, optional): An identifying tag for logging purposes.
            label_encoding (dict, optional): A dictionary with original labels as keys and encoded labels as values. If None, it defaults to index-based encoding of unique labels.
            download (bool, optional): If True, downloads the data if it does not exist locally. Default is False.
        """
        if data_path is None:
            data_path = get_data_path()
        self.data_path = data_path
        self.transform = torch.nn.Identity()
        if download:
            self.preprocess()
        if metadata is None:
            self.metadata = get_metadata(
                data_path,
                None,
                label_encoding,
                exclude=exclude,
                prescale=prescale,
                include_series=include_series,
                require_series=require_series,
            )
        else:
            self.metadata = metadata
        self.tag = tag
        mask_files = glob.glob(
            os.path.join(self.data_path, "Segmentation_Masks_NRRD/*/*Breast.seg*")
        )
        patients_ids_with_masks = [x.split("/")[-2] for x in mask_files]
        self.metadata = self.metadata[
            self.metadata.PatientID.str.contains("|".join(patients_ids_with_masks))
        ]
        self.load_boxes()

        studies = []
        for study in self.metadata.StudyInstanceUID.unique():
            series = self.metadata[self.metadata.StudyInstanceUID == study]
            patient_id = series.iloc[0].PatientID
            # could also stack different series instead of mixing them-- can make it configurable, let's see if mixing works first
            for path, series_description in zip(
                series.path.to_list(), series.CleanedSeriesDescription.to_list()
            ):
                data = {
                    "patient_id": patient_id,
                    "study_id": series.iloc[0].StudyInstanceUID,
                    "image": path,
                    "original_image": path,
                    "path": path,
                    "gt_mask": os.path.join(
                        self.data_path,
                        "Segmentation_Masks_NRRD",
                        patient_id,
                        f"Segmentation_{patient_id}_Breast.seg.nrrd",
                    ),
                    "series_description": [series_description],
                    "box": self.boxes[patient_id],
                }
                studies.append(data)
        studies = sorted(studies, key=lambda x: x["patient_id"])
        log_summary(self.tag, self.metadata, None)
        self.label_encoding = label_encoding
