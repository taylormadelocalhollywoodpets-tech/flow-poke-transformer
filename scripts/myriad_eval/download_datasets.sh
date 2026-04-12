# OWM
hf download CompVis/owm-95 --local-dir ./owm --repo-type dataset


# Physics
hf download CompVis/myriad-physics --local-dir ./physics --repo-type dataset

# Physion
wget https://physics-benchmarking-neurips2021-dataset.s3.amazonaws.com/Physion.zip
unzip -q Physion.zip -d ./physics/physion
mv ./physics/physion/Physion/* ./physics/physion
rm -rf Physion.zip ./physics/physion/__MACOSX ./physics/physion/Physion

# Physics IQ
git clone https://github.com/google-deepmind/physics-IQ-benchmark.git ./physics/physics-iq/piq_repo
cd ./physics/physics-iq
python ./piq_repo/code/download_physics_iq_data.py <<< 16
cd ../../
rm -rf ./physics/physics-iq/piq_repo
