#!/bin/bash
# run this after gene lists have been created using the
if [ $# != 2 ] ; then
  echo "Usage: `basename $0` {geneList} {outputDirectory}"
  echo "Searches ToppGene using a .csv file created by STANLY"
  exit 0;
fi
csvGeneList=$1
resultsDir=$2
# prepare output directory
if [ ! -d $resultsDir/toppGene ]; then
  mkdir -p $resultsDir/toppGene
fi
# get filename for naming outputs
filename=$(basename ${csvGeneList%%.csv})
txtGeneList=$resultsDir/toppGene/${filename}.txt
if [ ! -f $txtGeneList ]; then
  touch $txtGeneList
else
  rm $txtGeneList
  touch $txtGeneList
fi
while IFS=, read geneList nSigGenes; do
  echo -n "${geneList}," >> $txtGeneList
done < $csvGeneList

actList=$(cat $txtGeneList)
callCurl=$(echo "curl -H 'Content-Type: text/json' -d '{\"Symbols\":[\"")
for updatedList in ${actList//,/\",\"}; do
  callCurl="$callCurl$updatedList"
done
callCurl=$(echo "$callCurl\"]}' https://toppgene.cchmc.org/API/lookup")
eval $callCurl > ${resultsDir}/toppGene/${filename}_entrez_ouput.json

entrezList=`jq '.Genes[].Entrez' ${resultsDir}/toppGene/${filename}_entrez_ouput.json`
callApi=$(echo "curl -H 'Content-Type: text/json' -d '{\"Genes\":[")
for geneNumber in $entrezList; do
  callApi="${callApi}${geneNumber},"
done
callApi=${callApi%,*}
callApi=$(echo "${callApi}]}' https://toppgene.cchmc.org/API/enrich?pretty=true")
echo $callApi
eval $callApi > ${resultsDir}/toppGene/${filename}_func_enrich.json
