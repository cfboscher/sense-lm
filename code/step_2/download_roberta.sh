git clone https://huggingface.co/roberta-base

cd roberta-base
echo "Pulling with git lfs"


mv merges.txt merges-roberta-base.txt
mv tf_model.h5 roberta-base.h5
mv vocab.json vocab-roberta-base.json
mv config.json config-roberta-base.json
