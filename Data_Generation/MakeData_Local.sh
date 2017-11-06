for i in {0..365}
do
	echo $i
	python PyDataGen.py $i
done
