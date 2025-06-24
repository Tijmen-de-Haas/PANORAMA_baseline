#  Copyright 2024 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import SimpleITK as sitk
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

# imports required for running nnUNet algorithm
import subprocess
from subprocess import check_output, STDOUT, CalledProcessError
from pathlib import Path
import json
# imports required for my algorithm
from data_utils import resample_img, CropPancreasROI, GetFullSizDetectionMap, PostProcessing
import uuid

import warnings
warnings.filterwarnings("ignore")

def generate_uid():
    # Generate a UID with a root (usually 2.25 is a private root)
    # 2.25.xxx where xxx is a large integer derived from UUID
    uid = "2.25." + str(uuid.uuid4().int % (10**38))
    return uid

class PDACDetectionContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(UniqueImagesValidator(), UniquePathIndicesValidator())
            )
        )

        base_dir = Path.cwd()  # 👈 Get the folder this script is run in

        self.dicom_input_dir = Path("/DATA_INPUT")  # NEW: DICOM input
        

        self.nnunet_input_dir_lowres = Path("/opt/algorithm/nnunet/input_lowres")
        self.nnunet_input_dir_fullres = Path("/opt/algorithm/nnunet/input_fullres")
        self.nnunet_output_dir_lowres = Path("/opt/algorithm/nnunet/output_lowres")
        self.nnunet_output_dir_fullres = Path("/opt/algorithm/nnunet/output_fullres")
        self.nnunet_model_dir = Path("/opt/algorithm/nnunet/results")

        self.output_dir = Path("/DATA_OUTPUT")
        self.output_dir_images = self.output_dir / "dcm_images"
        

        for path in [
            self.nnunet_input_dir_lowres,
            self.nnunet_input_dir_fullres,
            self.nnunet_output_dir_lowres,
            self.nnunet_output_dir_fullres,
            self.output_dir,
            self.output_dir_images
        ]:
            path.mkdir(exist_ok=True, parents=True)
            

    def load_dicom_series(self):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(self.dicom_input_dir))
        print(len(dicom_names))
        reader.SetFileNames(dicom_names)
        reader.MetaDataDictionaryArrayUpdateOn() 
        image = reader.Execute()
        self.original_dicom_metadata = []
        for i, filename in enumerate(dicom_names):
            tags = reader.GetMetaDataKeys(i)
            meta = {tag: reader.GetMetaData(i, tag) for tag in tags}
            self.original_dicom_metadata.append(meta)
        return image

    def write_dicom_series(self, image, output_dir):
        image = sitk.Cast(image, sitk.sitkInt16)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()

        study_uid = self.original_dicom_metadata[0].get("0020|000D")
        if study_uid is None:
            print("Warning: StudyInstanceUID not found, generating new one.")
            study_uid = generate_uid()
        series_uid = generate_uid()

        for i in range(image.GetDepth()):
            slice_image = image[:, :, i]
            out_path = os.path.join(output_dir, f"{i:04d}.dcm")

            # Apply metadata from original slice if available
            for key in self.original_dicom_metadata[i].keys():
                value = self.original_dicom_metadata[i][key]
                slice_image.SetMetaData(key, value)


            slice_image.SetMetaData("0008|103e", "PDAC Detection Map")
            slice_image.SetMetaData("0020|000D", study_uid) 
            slice_image.SetMetaData("0020|000e", series_uid)

            writer.SetFileName(out_path)
            writer.Execute(slice_image)

    def process(self):
        itk_img = self.load_dicom_series()
        image_size = itk_img.GetSize()
        print(f"Image size: {image_size}") 

        # Step 1: Downsample image for low-res pancreas segmentation
        new_spacing = (4.5, 4.5, 9.0)
        image_resampled = resample_img(itk_img, new_spacing, is_label=False, out_size=[])
        sitk.WriteImage(image_resampled, str(self.nnunet_input_dir_lowres / "scan_0000.nii.gz"))

        # Step 2: Low-res pancreas mask prediction
        self.predict(
            input_dir=self.nnunet_input_dir_lowres,
            output_dir=self.nnunet_output_dir_lowres,
            task="Dataset103_PANORAMA_baseline_Pancreas_Segmentation"
        )
        mask_low_res = sitk.ReadImage(str(self.nnunet_output_dir_lowres / "scan.nii.gz"))

        # Step 3: Crop image around pancreas
        cropped_image, crop_coordinates = CropPancreasROI(itk_img, mask_low_res, [100, 50, 15])
        sitk.WriteImage(cropped_image, str(self.nnunet_input_dir_fullres / "scan_0000.nii.gz"))

        # Step 4: Full-res detection
        self.predict(
            input_dir=self.nnunet_input_dir_fullres,
            output_dir=self.nnunet_output_dir_fullres,
            task="Dataset104_PANORAMA_baseline_PDAC_Detection",
            trainer="nnUNetTrainer_Loss_CE_checkpoints",
            checkpoint='checkpoint_best_panorama.pth'
        )

        pred_npz = np.load(str(self.nnunet_output_dir_fullres / "scan.npz"))
        pred_nifti = str(self.nnunet_output_dir_fullres / "scan.nii.gz")
        prediction_postprocessed = PostProcessing(pred_npz, pred_nifti)
        detection_map, patient_level_prediction = GetFullSizDetectionMap(
            prediction_postprocessed, crop_coordinates, itk_img
        )

        # NEW: Write detection map as DICOM series
        self.write_dicom_series(detection_map, self.output_dir_images)

        write_json_file(location=self.output_dir / "pdac-likelihood.json", content=patient_level_prediction)



    def predict(self, input_dir, output_dir, task="Task103_AllStructures", trainer="nnUNetTrainer",
                    configuration="3d_fullres", checkpoint="checkpoint_final.pth", folds="0,1,2,3,4", 
                    store_probability_maps=True):
            """
            Use trained nnUNet network to generate segmentation masks
            """

            # Set environment variables
            os.environ['RESULTS_FOLDER'] = str(self.nnunet_model_dir)

            # Run prediction script
            cmd = [
                'nnUNetv2_predict',
                '-d', task,
                '-i', str(input_dir),
                '-o', str(output_dir),
                '-c', configuration,
                '-tr', trainer,
                '--disable_progress_bar',
                '--continue_prediction'
            ]

            if folds:
                cmd.append('-f')
                cmd.extend(folds.split(','))

            if checkpoint:
                cmd.append('-chk')
                cmd.append(checkpoint)

            if store_probability_maps:
                cmd.append('--save_probabilities')

            cmd_str = " ".join(cmd)
            subprocess.check_call(cmd_str, shell=True)

def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))

if __name__ == "__main__":
    PDACDetectionContainer().process()
