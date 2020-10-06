#!/bin/bash

echo "Processing CoLA"
echo "==============="
cd CoLA
cat train.tsv | cut -f4 > train.word
cat train.tsv | cut -f2 > train.label

cat dev.tsv | cut -f4 > dev.word
cat dev.tsv | cut -f2 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..

echo "Processing SST-2"
echo "================"
cd SST-2
tail -n +2 train.tsv | cut -f1 > train.word
tail -n +2 train.tsv | cut -f2 > train.label

tail -n +2 dev.tsv | cut -f1 > dev.word
tail -n +2 dev.tsv | cut -f2 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..

echo "Processing MRPC"
echo "==============="
cd MRPC
tail -n +2 train.tsv | cut -f4,5 | sed 's/\t/ \|\|\| /g' > train.word
tail -n +2 train.tsv | cut -f1 > train.label

tail -n +2 dev.tsv | cut -f4,5 | sed 's/\t/ \|\|\| /g' > dev.word
tail -n +2 dev.tsv | cut -f1 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..

echo "Processing QQP"
echo "=============="
cd QQP
tail -n +2 train.tsv | cut -f4,5 | sed 's/\t/ \|\|\| /g' > train.word
tail -n +2 train.tsv | cut -f6 > train.label

tail -n +2 dev.tsv | cut -f4,5 | sed 's/\t/ \|\|\| /g' > dev.word
tail -n +2 dev.tsv | cut -f6 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..

echo "Processing STS"
echo "=============="
cd STS-B
tail -n +2 train.tsv | cut -f8,9 | sed 's/\t/ \|\|\| /g' > train.word
tail -n +2 train.tsv | cut -f10 > train.label

tail -n +2 dev.tsv | cut -f8,9 | sed 's/\t/ \|\|\| /g' > dev.word
tail -n +2 dev.tsv | cut -f10 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..

echo "Processing MNLI"
echo "==============="
cd MNLI
tail -n +2 train.tsv | cut -f9,10 | sed 's/\t/ \|\|\| /g' > train.word
tail -n +2 train.tsv | cut -f12 > train.label

tail -n +2 dev_matched.tsv | cut -f9,10 | sed 's/\t/ \|\|\| /g' > dev.word
tail -n +2 dev_matched.tsv | cut -f16 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev_matched.tsv dev.word dev.label
cd ..

echo "Processing QNLI"
echo "==============="
cd QNLI
tail -n +2 train.tsv | cut -f2,3 | sed 's/\t/ \|\|\| /g' > train.word
tail -n +2 train.tsv | cut -f4 > train.label

tail -n +2 dev.tsv | cut -f2,3 | sed 's/\t/ \|\|\| /g' > dev.word
tail -n +2 dev.tsv | cut -f4 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..

echo "Processing RTE"
echo "=============="
cd RTE
tail -n +2 train.tsv | cut -f2,3 | sed 's/\t/ \|\|\| /g' > train.word
tail -n +2 train.tsv | cut -f4 > train.label

tail -n +2 dev.tsv | cut -f2,3 | sed 's/\t/ \|\|\| /g' > dev.word
tail -n +2 dev.tsv | cut -f4 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..

echo "Processing WNLI"
echo "==============+"
cd WNLI
tail -n +2 train.tsv | cut -f2,3 | sed 's/\t/ \|\|\| /g' > train.word
tail -n +2 train.tsv | cut -f4 > train.label

tail -n +2 dev.tsv | cut -f2,3 | sed 's/\t/ \|\|\| /g' > dev.word
tail -n +2 dev.tsv | cut -f4 > dev.label

wc -l train.tsv train.word train.label 
wc -l dev.tsv dev.word dev.label
cd ..
