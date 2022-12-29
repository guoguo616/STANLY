#!/bin/bash
# run this after gene lists have been created using the
resultsDir=../derivatives/221224

if [ ! -d $resultsDir/toppGene ]; then
  mkdir $resultsDir/toppGene
fi
csvGeneList=$resultsDir/listOfSigSleepDepGenes20221224-162709.csv
txtGeneList=$resultsDir/toppGene/geneListForToppGene.txt
if [ ! -f $txtGeneList ]; then 
  touch $txtGeneList
  while IFS=, read geneList nSigGenes; do
    # echo $geneList
    echo -n "${geneList}," >> $txtGeneList
  done < $csvGeneList
fi
# for txtFile in $resultsDir/toppGene/geneListForToppGene.txt; do
listName=$(echo $txtGeneList | cut -d / -f 4 | cut -d . -f 1)
actList=$(cat $txtGeneList)
callCurl=$(echo "curl -H 'Content-Type: text/json' -d '{\"Symbols\":[\"")
for updatedList in ${actList//,/\",\"}; do
  callCurl="$callCurl$updatedList"
done
callCurl=$(echo "$callCurl\"]}' https://toppgene.cchmc.org/API/lookup")
eval $callCurl > ${resultsDir}/${listName}_entrez_ouput.json

entrezList=`jq '.Genes[].Entrez' ${resultsDir}/${listName}_entrez_ouput.json`
echo $entrezList
callApi=$(echo "curl -H 'Content-Type: text/json' -d '{\"Genes\":[")
for geneNumber in $entrezList; do
  callApi="${callApi}${geneNumber},"
done
callApi=${callApi%,*}
callApi=$(echo "${callApi}]}' https://toppgene.cchmc.org/API/enrich?pretty=true")
eval $callApi > ${resultsDir}/toppGene/${listName}_func_enrich.json
# done
