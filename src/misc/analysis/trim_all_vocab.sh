dir=`find //dfs/scratch0/maxlam/w2bfinalvectors/full_vectors -name "*txt"`
outdir=/dfs/scratch0/maxlam/w2bfinalvectors/trimmed_vectors/
trim_level=400000

for f in ${dir}
do
    name=`echo $f | grep -oP "mikolov.*"`
    echo "python trim_vocabulary.py ${f} ${trim_level} /dfs/scratch1/senwu/embedding_evaluation/data/wikimedia/20171204/wiki.en.txt ${outdir}/$name"
    python trim_vocabulary.py ${f} ${trim_level} /dfs/scratch1/senwu/embedding_evaluation/data/wikimedia/20171204/wiki.en.txt ${outdir}/$name
done
