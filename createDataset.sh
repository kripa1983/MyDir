awk '{print$1}' trec678-qrel|sort |uniq > qnos.txt

head -1 qnos.txt > qnos1.txt

rm -rf rel.txt
rm -rf nrel.txt

export PATHTOINDRI=/home/kripa/ir/tools/indri-5.11

CURRENT_DIR=$(pwd)

cd $PATHTOINDRI

coll_size=$(dumpindex ./TREC678_index-StopwordsRemoved-PorterStemmed/ s|tail -n +2|awk '{print$2}'|head -1)
total_terms=$(dumpindex ./TREC678_index-StopwordsRemoved-PorterStemmed/ s|tail -n +4|awk '{print$3}'|head -1)
avg_doc_len=$(echo "$total_terms/$coll_size"|bc)

echo "Coll Size="$coll_size" Avg. Doc Len="$avg_doc_len

cd $CURRENT_DIR

gcc -o BM25 BM25.c -lm

k=25 #Top k terms in a document

while read f;
do

	awk -v var=$f '$1==var&&$4==1{print "1	"$3}' trec678-qrel > rel.txt
	awk -v var=$f '$1==var&&$4==0{print "0	"$3}' trec678-qrel > nrel.txt

	rel_num=$(cat rel.txt|wc -l)
	nrel_num=$(cat nrel.txt|wc -l)

	rel_choice=$(echo "$rel_num*3/4"|bc)  #3/4 of all the rel docs will be chosen

	#echo $rel_choice

	cat rel.txt | head -$rel_choice > training_docid.txt
	cat nrel.txt | head -$rel_choice >> training_docid.txt #training set containd 50% rel and 50% nrel docs

	rest_num=`expr $rel_choice + 1`

	cat rel.txt | tail -n +$rest_num > test_docid.txt
	cat nrel.txt | tail -n +$rest_num >> test_docid.txt # remaining docs form the test set


	head -1 training_docid.txt > t


	# Creating the training file

	rm -rf training-LabelDocidDocumentTerms.txt

	while read f1;
	do
		
		rel_label=$(echo $f1|awk '{print$1}')
		dno=$(echo $f1|awk '{print$2}')
		cd $PATHTOINDRI
		did=$(dumpindex ./TREC678_index-StopwordsRemoved-PorterStemmed/ di docno $dno)
		doc_len=$(dumpindex ./TREC678_index-StopwordsRemoved-PorterStemmed/ dv $did|tail -n +3|awk '{print$3}'|sed 's/\[OOV\]/#/g'|egrep -v "#"|wc -l)
		dumpindex ./TREC678_index-StopwordsRemoved-PorterStemmed/ dv $did|tail -n +3|awk '{print$3}'|sed 's/\[OOV\]//g'|sort|uniq -c|tail -n +2|awk '{print$2" "$1}'|sort > vec.txt
		dumpindex ./TREC678_index-StopwordsRemoved-PorterStemmed/ v|tail -n +2|awk '{print$1" "$3}'|sort > voc.txt

		join vec.txt voc.txt > join_res.txt

		cp join_res.txt $CURRENT_DIR

		cd $CURRENT_DIR

		echo "Doc Len="$doc_len

		rm -rf term-bm25idf.txt

		

			

		./BM25 join_res.txt $doc_len $avg_doc_len $coll_size > term-bm25idf.txt

			

		top_terms=$(sort -k 2,2nr term-bm25idf.txt|awk '{print$1}'|head -$k|tr "\n" " ")

		echo $rel_label" "$dno" "$top_terms >> training-LabelDocidDocumentTerms.txt #printing in the input format of "readEmbedding.py"
		

	done < training_docid.txt #t

	
	

done < qnos1.txt


