#!/bin/bash

function setup_proxy() {
    export http_proxy=http://10.132.19.35:7890
    export https_proxy=http://10.132.19.35:7890
}

function unset_proxy() {
    unset http_proxy
    unset https_proxy
}

# 更新驱动
function update_driver() {
    wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.18.0/habanalabs-installer.sh
    chmod +x habanalabs-installer.sh
    ./habanalabs-installer.sh upgrade --type base --verbose
    ./habanalabs-installer.sh upgrade --type dependencies --verbose
    ./habanalabs-installer.sh upgrade --type pytorch --verbose
}

function download_dataset() {
    echo "=> download dataset..."
    dataset_url="https://cloud.tsinghua.edu.cn/seafhttp/files/b373a591-f999-485c-b5c0-e3abffa371ed/AdvertiseGen.tar.gz"
    wget -O AdvertiseGen.tar.gz $dataset_url
    tar -xvzf AdvertiseGen.tar.gz -C /data
    echo "=> download dataset done."
}

function download_model() {
    echo "=> downloading model..."
    mkdir -p /data/chatglm3-6b
    modelscope download --model ZhipuAI/chatglm3-6b --local_dir /data/chatglm3-6b
    cp models/tokenization_chatglm.py /data/chatglm3-6b
    echo "=> downloading model done..."
}

function install_pypi() {
    pip install -r requirements.txt
    setup_proxy
    pip install git+https://github.com/HabanaAI/vllm-hpu-extension.git@250622e752917ab4d35131ac85ab1f6eef8043a9
    unset_proxy
}

function main() {
    update_driver
    install_pypi
    download_model
    download_dataset
}

main $@